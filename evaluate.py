import json
import os

from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation() -> dict:
    """
    Run the classifier against the held-out test set and return full results.
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(episode["description"], labeled_examples)

        predicted_label = prediction.get("label", "unknown")
        ground_truth_label = episode["label"]

        results.append(
            {
                "id": episode["id"],
                "title": episode["title"],
                "description": episode["description"],
                "ground_truth": ground_truth_label,
                "predicted": predicted_label,
                "reasoning": prediction.get("reasoning", ""),
                "confidence": prediction.get("confidence", 0),
                "correct": predicted_label == ground_truth_label,
            }
        )

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.

    Accuracy = correct predictions / total predictions.
    """
    if not predictions or not ground_truth:
        return 0.0

    total = min(len(predictions), len(ground_truth))
    if total == 0:
        return 0.0

    correct = 0
    for predicted, truth in zip(predictions, ground_truth):
        if predicted == truth:
            correct += 1

    return correct / total


def compute_per_class_accuracy(predictions: list[str], ground_truth: list[str]) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.

    For each label in VALID_LABELS, compute:
    - correct: number of episodes with this ground-truth label predicted correctly
    - total: number of episodes with this ground-truth label
    - accuracy: correct / total, or 0.0 if total is 0
    """
    stats = {
        label: {"correct": 0, "total": 0, "accuracy": 0.0}
        for label in VALID_LABELS
    }

    for predicted, truth in zip(predictions, ground_truth):
        if truth not in stats:
            continue

        stats[truth]["total"] += 1
        if predicted == truth:
            stats[truth]["correct"] += 1

    for label in VALID_LABELS:
        total = stats[label]["total"]
        if total > 0:
            stats[label]["accuracy"] = stats[label]["correct"] / total
        else:
            stats[label]["accuracy"] = 0.0

    return stats


def compute_average_confidence_per_class(results: list[dict]) -> dict[str, float]:
    """
    Optional challenge: average the classifier confidence by ground-truth class.
    """
    buckets = {label: [] for label in VALID_LABELS}

    for result in results:
        truth = result.get("ground_truth")
        confidence = result.get("confidence", 0)
        if truth in buckets:
            try:
                buckets[truth].append(float(confidence))
            except (TypeError, ValueError):
                buckets[truth].append(0.0)

    averages = {}
    for label, values in buckets.items():
        averages[label] = sum(values) / len(values) if values else 0.0

    return averages


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)
    confidence_by_class = compute_average_confidence_per_class(results)

    lines = [
        "## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]

    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (10 - int(stats["accuracy"] * 10))
        avg_confidence = confidence_by_class.get(label, 0.0)
        lines.append(
            f" {label:<12} {bar} {stats['accuracy']:.0%} "
            f"({stats['correct']}/{stats['total']}) | avg confidence: {avg_confidence:.1f}/10"
        )

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(
                f" [{r['ground_truth']} → {r['predicted']}] {r['title']} "
                f"(confidence: {r.get('confidence', 0)}/10)"
            )
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)
