# Classifier Spec — Completed

## build_few_shot_prompt(labeled_examples, description)

### Task instruction
The LLM should classify podcast episode descriptions by structural format, not by topic, mood, or marketing language. It must choose exactly one of four valid labels:

- `interview`: host speaks with one or more guests in a Q&A or conversation format.
- `solo`: one host speaks alone from memory, personal experience, opinion, or analysis.
- `panel`: three or more speakers discuss a topic together as rough equals.
- `narrative`: reported or documentary-style story assembled from external sources.

### Labeled example format
Each labeled example is formatted as:

```text
Title: {title}
Description: {description}
Label: {label}
```

Examples are separated with `---` so the model can clearly see where one example ends and another begins.

### New episode format
The new episode is presented after the labeled examples:

```text
New episode to classify:
Description: {description}
```

The classifier UI only passes the description, so the prompt does not require a title for the new episode.

### Output format
Use JSON only:

```json
{
  "label": "interview | solo | panel | narrative",
  "confidence": 0,
  "reasoning": "brief reason using the episode structure"
}
```

Reason: JSON is easier to parse than free-form text. The code still includes fallback parsing in case the model returns `Label: X` instead.

### Edge cases
If `labeled_examples` is empty, the prompt still includes the taxonomy definitions and tells the model to classify using only those definitions. If the description is short, the model is instructed to classify based on structural clues and return its best valid label. If no valid label can be parsed, the code returns `unknown` instead of crashing.

---

## classify_episode(description, labeled_examples)

### Step 1 — Build the prompt
Call `build_few_shot_prompt(labeled_examples, description)`.

### Step 2 — Send to the LLM
Call `_client.chat.completions.create()` using:

- `model=LLM_MODEL`
- `messages=[{"role": "user", "content": prompt}]`
- `temperature=0`
- `max_tokens=300`

Then read `response.choices[0].message.content`.

### Step 3 — Parse the response
First try `json.loads(response_text)`. If that fails, search the response for a JSON object using a regex and parse that. If that also fails, use a fallback parser for formats like:

```text
Label: interview
Reasoning: The host speaks with a named guest.
```

### Step 4 — Validate the label
Normalize the label by stripping whitespace and punctuation and converting to lowercase. If the label is not exactly one of `VALID_LABELS`, return `unknown`.

### Step 5 — Handle errors gracefully
If the API call fails, the response is unparseable, or the description is empty, return:

```python
{
    "label": "unknown",
    "reasoning": "explanation of the problem",
    "confidence": 0,
}
```

This prevents one bad LLM response from crashing the full 20-episode evaluation.

---

## Implementation Notes

### Test: what does the raw LLM response look like for one episode?

```text
Episode tested: The Aral Sea: A Disaster in Four Acts
Raw response text:
{"label":"narrative","confidence":9,"reasoning":"The episode tells a structured story across time about an environmental disaster rather than presenting a Q&A, solo reflection, or panel discussion."}
```

### How did you parse the label out of the response?
I used `json.loads()` first. Then I normalized the `label` value with `strip()` and `lower()`, removed punctuation/markdown characters, and checked whether it was in `VALID_LABELS`. If JSON parsing fails, the code falls back to regex/string parsing.

### Did any episodes return `unknown`?
No, not when the model followed the requested JSON format. If `unknown` appears, the first thing to check is the raw response text because the model may have returned a valid label in an unexpected format.

### One thing about the output format that surprised you
Even with clear instructions, LLMs can sometimes add extra text or markdown. That is why the code uses both strict JSON parsing and fallback parsing.
