# Evaluation Spec — Completed

## compute_accuracy(predictions, ground_truth)

### Formula
Accuracy equals the number of predictions that exactly match the ground-truth labels divided by the total number of predictions.

### Step-by-step logic
1. If either list is empty, return `0.0`.
2. Pair predictions and ground-truth labels with `zip()`.
3. Count how many pairs are equal.
4. Divide the correct count by the total number of paired items.
5. Return the result as a float.

### Edge case — both lists are empty
Return `0.0` because there are no examples to evaluate. This avoids division by zero.

### Worked example
```python
predictions = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo", "narrative"]
```

Correct predictions:

1. `interview == interview` → correct
2. `solo == solo` → correct
3. `panel != solo` → incorrect
4. `interview != narrative` → incorrect

Result: `2 / 4 = 0.5`

---

## compute_per_class_accuracy(predictions, ground_truth)

### What does correct mean for a class?
For a class like `interview`, an episode counts as correct only when the ground-truth label is `interview` and the predicted label is also `interview`.

### What does total mean for a class?
Total means the number of test episodes whose ground-truth label is that class. It is not the total number of predictions overall.

### Step-by-step logic
1. Initialize a dictionary for every label in `VALID_LABELS` with `correct=0`, `total=0`, and `accuracy=0.0`.
2. Loop over each `(predicted, truth)` pair.
3. Add one to `total` for the ground-truth label.
4. If `predicted == truth`, add one to `correct` for that class.
5. After the loop, compute `accuracy = correct / total` for each class.
6. If a class has `total == 0`, leave accuracy as `0.0`.
7. Return the stats dictionary.

### Edge case — class has no examples
If a class has no examples in `ground_truth`, its accuracy should be `0.0`. This avoids dividing by zero and clearly shows there was no data for that class.

### Worked example
```python
predictions = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo", "solo", "panel", "narrative"]
```

| label | correct | total | accuracy |
|---|---:|---:|---:|
| interview | 1 | 1 | 1.0 |
| solo | 1 | 2 | 0.5 |
| panel | 1 | 1 | 1.0 |
| narrative | 0 | 1 | 0.0 |

---

## Reflection questions

### 1. Why is per-class accuracy more informative than overall accuracy alone?
Overall accuracy can hide class-specific failure. A classifier could score well overall by doing well on common or easy classes while failing badly on one label. Per-class accuracy shows which categories are actually working.

### 2. If `panel` episodes are consistently misclassified as `interview`, what does that tell you?
It suggests the prompt or training examples are not making the panel/interview boundary clear enough. The model may be treating any multi-person conversation as an interview unless the prompt emphasizes equal-standing group discussion.

### 3. What if there were 100 training episodes or 200 test episodes?
More high-quality training examples would likely improve the few-shot signal, especially for ambiguous classes. More test episodes would make the evaluation more reliable because each per-class score would be based on more examples.
