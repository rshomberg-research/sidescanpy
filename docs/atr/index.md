# Two-stage target recognition, step by step

Six scripts that train and test a two-stage recognizer: a detector that
finds anything on the seabed in whole images, then a classifier that
looks closely at each detection. They train on a public labelled image
set, test on a year of data the models never saw, show how a person
reviews the results, and finally run on sidescanpy swath tiles with
positions. [Target recognition](../atr.md) shows how these steps fit
with converting, correcting and tiling your own data.

The scripts are in `examples/atr/` and need packages that are not
sidescanpy dependencies (Ultralytics YOLO, which brings PyTorch and
torchvision, and FiftyOne for step 5) and a GPU. They are not run when
this site is built: each page shows the results of a run on a
workstation with one 20 GB GPU. Run them in order with the same working
folder, e.g. `python examples/atr/atr_1_prepare.py atr_santos`.

1. **[Labelled sonar images](prepare.md).** Downloads 1170 sidescan
   images with 668 labelled objects (mine-like contacts and other bottom
   objects) and splits them by year into training, validation and test
   sets, because neighbouring images overlap.
2. **[Train the detector](detector.md).** Trains Ultralytics YOLO to find
   one class, "object". It is judged on how few objects it misses: false
   alarms are left for the classifier.
3. **[Train the classifier](classifier.md).** Fine-tunes a small
   ResNet-18 on chips around each object, with the detector's own false
   alarms as the "clutter" class. Using random seabed as clutter instead
   barely removed any false alarms.
4. **[Test both stages](evaluate.md).** Runs detector and classifier on
   the held-out year and compares the result with the detector alone:
   the classifier halves the false alarms for a few lost hits.
5. **[Review and correct the labels](review.md).** Loads images, labels
   and predictions into FiftyOne, sends the mistakes to CVAT for
   correction, and exports the corrected labels as COCO for the next
   training round.
6. **[Run the models on a survey](survey.md).** Cuts swath tiles from a
   corrected public Klein line, turns the detections into targets with
   positions, and scores them against a catalog in meters.

```{toctree}
:hidden:

prepare
detector
classifier
evaluate
review
survey
```
