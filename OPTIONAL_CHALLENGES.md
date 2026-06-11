# Optional Challenges — Completed Notes

## 1. Find the breaking point
To simulate a smaller few-shot training set, keep only 3 examples per class in `data/my_labels.json` and set the rest to `null`. Then rerun evaluation.

Expected observation:
- Accuracy will usually drop when the model sees fewer examples.
- `panel` and `narrative` are likely to degrade first because they depend on structural boundaries that can be confused with interview or solo.
- Few-shot classification becomes unreliable when examples no longer show the center of each class and at least one edge case.

Example write-up:
> When I reduced the training set to 3 examples per class, the classifier became more sensitive to the exact examples shown in the prompt. Interview and solo stayed relatively stable because their structure is obvious, but panel and narrative were more fragile. Panel can be confused with interview if the description sounds like a conversation, and narrative can be confused with solo if the story is first-person. This suggests that few-shot prompting needs enough examples to show not only the obvious cases, but also the boundaries between classes.

## 2. Tune the prompt systematically
Try these three versions and compare per-class accuracy:

1. **Balanced order:** group examples by label with the same number per class.
2. **Boundary-first examples:** put confusing examples near each other, such as solo vs narrative and interview vs panel.
3. **More taxonomy context:** include the edge-case rules before the examples.

Example write-up:
> The model is sensitive to how examples are ordered and explained. The best-performing prompt was the one that included the taxonomy and edge-case rules before the examples. Grouping examples by label made the pattern easier to see, but placing boundary cases near each other helped the model distinguish similar formats. The biggest improvement came from explicitly saying to classify by structure, not topic.

## 3. Add a confidence score
Completed in the included code:

- `classify_episode()` now asks the LLM for a `confidence` value from 0 to 10.
- `run_evaluation()` stores confidence for each result.
- `format_evaluation_report()` reports average confidence by class.

Expected observation:
> Low confidence often appears near ambiguous cases. If the model is confident and wrong, the issue is probably the prompt or the labeled examples. If the model is low-confidence and wrong, the episode itself may be structurally ambiguous.

## 4. Write an adversarial description

### Artificially confusing version
> In this episode, I sit down with four experts to tell the story of a vanished neighborhood. Their voices appear throughout, but I guide the listener through the timeline from the first eviction notice to the final demolition.

Correct label: `narrative`

Why it is tricky: It mentions “sit down with four experts,” which sounds like interview or panel, but the structure is a guided reported story built from voices and a timeline.

### More human-hard version
> After my father died, I found a box of cassette tapes he recorded over twenty years. This episode is my attempt to understand the person he was before I knew him, using those tapes, old family letters, and conversations with my aunt.

Correct label: `narrative`

Why it is tricky: It is first-person and emotional, so a person might label it `solo`, but the story is assembled from external sources: tapes, letters, and another person's memories.

Example write-up:
> The classifier handled the obviously adversarial description better when the prompt included the edge-case rule about narratives built from interviews. The subtler family-history example was harder because it sounds like a personal reflection. The deciding factor is source material: if the host is only speaking from memory, it is solo; if the host builds the episode from documents, tapes, letters, or other people’s interviews, it is narrative.
