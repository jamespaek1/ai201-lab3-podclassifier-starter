from __future__ import annotations

import json
import os
import re
from typing import Any

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE


_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
    - "id" : episode ID
    - "title" : episode title
    - "podcast" : podcast name
    - "description" : episode description
    - "label" : the label from my_labels.json

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled examples.
    """
    label_definitions = """
Valid labels:
- interview: a host speaks with one or more guests in a Q&A or conversation format. A clear host-guest dynamic is present.
- solo: one host speaks alone from memory, personal experience, opinion, or analysis. No guests or assembled external sources drive the structure.
- panel: three or more speakers discuss a topic together as rough equals. No single person is the main interview subject.
- narrative: a reported or documentary-style story assembled from external sources such as interviews, archives, documents, recordings, or field reporting.
""".strip()

    example_blocks = []
    for example in labeled_examples:
        example_blocks.append(
            "\n".join(
                [
                    f"Title: {example.get('title', '').strip()}",
                    f"Description: {example.get('description', '').strip()}",
                    f"Label: {example.get('label', '').strip()}",
                ]
            )
        )

    if example_blocks:
        examples_text = "\n\n---\n\n".join(example_blocks)
    else:
        examples_text = "No labeled examples were provided. Use only the taxonomy definitions."

    return f"""
You are a careful podcast-format classifier.

Classify the new podcast episode description into exactly one of these labels:
{", ".join(VALID_LABELS)}

Classify by STRUCTURE, not topic, mood, or marketing language.

{label_definitions}

Important edge-case rules:
- If an episode is Q&A with a guest, classify it as interview even if the guest tells a dramatic story.
- If one host tells a first-person personal story from memory or opinion, classify it as solo.
- If three or more speakers discuss as rough equals, classify it as panel.
- If the episode reconstructs events from external materials, reporting, documents, archives, or interview excerpts, classify it as narrative.

Labeled examples:
---
{examples_text}
---

New episode to classify:
Description: {description.strip()}

Return ONLY a valid JSON object with this exact shape:
{{
  "label": "interview | solo | panel | narrative",
  "confidence": 0,
  "reasoning": "brief reason using the episode structure"
}}

Rules for your response:
- The label value must be exactly one of: {", ".join(VALID_LABELS)}.
- confidence must be an integer from 0 to 10.
- Do not include markdown fences.
- Do not include any text before or after the JSON.
""".strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try to parse the first JSON object in an LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_label(label_text: Any) -> str:
    """Normalize possible label text and validate it against VALID_LABELS."""
    if not isinstance(label_text, str):
        return "unknown"

    cleaned = label_text.strip().lower()
    cleaned = cleaned.strip("`*_ .,:;!?\"'")

    if cleaned.startswith("label:"):
        cleaned = cleaned.split(":", 1)[1].strip()

    # Handles responses like "interview - because..." or "interview\nbecause..."
    first_token = re.split(r"[\s\n\r\-:]+", cleaned)[0]

    if cleaned in VALID_LABELS:
        return cleaned
    if first_token in VALID_LABELS:
        return first_token

    # Last-resort fallback: if a valid label appears as a standalone word.
    for valid_label in VALID_LABELS:
        if re.search(rf"\b{re.escape(valid_label)}\b", cleaned):
            return valid_label

    return "unknown"


def _extract_confidence(value: Any) -> int:
    """Return confidence as an integer between 0 and 10."""
    try:
        confidence = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, confidence))


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns:
        {
            "label": one of VALID_LABELS or "unknown",
            "reasoning": brief explanation,
            "confidence": integer 0-10
        }
    """
    if not description or not description.strip():
        return {
            "label": "unknown",
            "reasoning": "No episode description was provided.",
            "confidence": 0,
        }

    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        response_text = response.choices[0].message.content.strip()
        parsed = _extract_json_object(response_text)

        if parsed is not None:
            label = _normalize_label(parsed.get("label"))
            reasoning = str(parsed.get("reasoning", "")).strip()
            confidence = _extract_confidence(parsed.get("confidence", 0))
        else:
            # Fallback for non-JSON responses like:
            # Label: interview
            # Reasoning: The host speaks with a named guest.
            label_match = re.search(r"label\s*:\s*([^\n]+)", response_text, flags=re.IGNORECASE)
            reasoning_match = re.search(r"reasoning\s*:\s*(.+)", response_text, flags=re.IGNORECASE | re.DOTALL)

            label_source = label_match.group(1) if label_match else response_text.splitlines()[0]
            label = _normalize_label(label_source)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text
            confidence = 0

        if label not in VALID_LABELS:
            return {
                "label": "unknown",
                "reasoning": f"Could not parse a valid label from the model response: {response_text}",
                "confidence": 0,
            }

        if not reasoning:
            reasoning = "The model returned a valid label but did not provide reasoning."

        return {
            "label": label,
            "reasoning": reasoning,
            "confidence": confidence,
        }

    except Exception as error:
        return {
            "label": "unknown",
            "reasoning": f"Classifier error: {error}",
            "confidence": 0,
        }
