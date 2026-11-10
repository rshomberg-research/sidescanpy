# 2. Train the detector

The first stage finds *something* on the seabed. It is
[Ultralytics YOLO](https://docs.ultralytics.com/) (YOLO11s, from
weights pretrained on everyday photographs), trained on the one-class
set from [step 1](prepare.md). Whatever it misses is lost for good,
while its false alarms are the classifier's job, so it is judged mainly
on recall and later run with a low confidence threshold.

Augmentation flips left to right (port and starboard swap, which is
fine for these one-sided images) but never up and down: a shadow always
falls away from the sonar.

## Result

55 epochs in 6 minutes; early stopping kept epoch 25. On the test year:

```text
test (2017): precision 0.39, recall 0.57, mAP50 0.33, mAP50-95 0.15
```

```{image} ../assets/atr/detector_training.png
:alt: YOLO training and validation losses, precision, recall and mAP per epoch
```

Modest, from 589 training boxes. Training at the images' full 1024 px
did worse (mAP50 0.02 on the test year): the validation year is small,
and early stopping picked an epoch that did not generalize. More
labelled data from more surveys helps more than any setting.

```{literalinclude} ../../examples/atr/atr_2_train_detector.py
:language: python
```
