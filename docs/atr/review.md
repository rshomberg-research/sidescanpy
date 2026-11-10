# 5. Review and correct the labels

A person checks what the two stages found before it counts: confirms
objects, rejects false alarms, fixes classes and adds what was missed.
The corrected labels are the training data for the next round, which is
how a small labelled set grows.

[FiftyOne](https://docs.voxel51.com/) shows images with their labels
and predictions, filters them (low-confidence calls, every `NOMBO`, the
images with a miss), and hands images to an annotation tool such as
CVAT for corrections. The script:

- loads the test images with their labels and the two-stage
  predictions;
- scores the predictions inside FiftyOne, so the app shows true
  positives, false positives and misses per image;
- sends every image with a false positive or a miss to CVAT, and loads
  the corrections back;
- exports the corrected labels as COCO, which sidescanpy reads with
  `export.ml.read_coco_boxes` and turns into targets on the sonar data
  with `export.ml.boxes_to_targets`.

It opens an app and waits for a person, so it has no recorded output.
It needs `pip install fiftyone` and a CVAT account.

```{literalinclude} ../../examples/atr/atr_5_review_fiftyone.py
:language: python
```
