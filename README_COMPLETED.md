# Pod Classifier Completed Files

Copy these files into your VS Code repo:

- `data/my_labels.json`
- `classifier.py`
- `evaluate.py`
- `specs/classifier-spec.md`
- `specs/evaluation-spec.md`
- `OPTIONAL_CHALLENGES.md`

Then run:

```bash
python -c "from classifier import load_labeled_examples; examples = load_labeled_examples(); print(f'{len(examples)} labeled examples loaded'); print([e['label'] for e in examples])"
python app.py
```

In the app:

1. Go to **Classify**.
2. Test **The Aral Sea: A Disaster in Four Acts**. It should classify as `narrative`.
3. Test **The Case for Four-Day Workweeks**. It should classify as `solo`.
4. Go to **Evaluate** and click **Run Evaluation**.

Note: You must have a valid `GROQ_API_KEY` in your `.env` file for the classifier/evaluation to run.
