# 4. Test both stages

The year 2017 was kept out of training and validation. Both stages run
on it as they would on a new survey: the detector with a low confidence
threshold (0.05), then the classifier on a chip around each detection,
dropping what it calls `clutter`. A detection is a hit when it overlaps
a labelled object by at least 10 % (intersection over union); the
objects are a few pixels across, so a stricter overlap punishes small
offsets more than real mistakes.

## Result

30 labelled objects in the test year:

| | Hits | False alarms | Misses | Precision | Recall |
|---|---|---|---|---|---|
| Detector alone, threshold 0.25 | 4 | 5 | 26 | 0.44 | 0.13 |
| Detector alone, threshold 0.05 | 20 | 57 | 10 | 0.26 | 0.67 |
| Detector (0.05) + classifier | 15 | 26 | 15 | 0.37 | 0.50 |

The low threshold finds two thirds of the objects but raises 57 false
alarms; the classifier removes more than half of them and loses five
hits. With 30 test objects these numbers move a lot from run to run;
the comparison between rows is the point, not the values.

```{image} ../assets/atr/two_stage_test.jpg
:alt: Six test images with labelled objects in green and two-stage detections in red and cyan
```

Labelled objects in green, two-stage calls in red (`MILCO`) and cyan
(`NOMBO`). On sonar data with positions the same comparison is made in
meters with `targets.match_targets` ([step 6](survey.md)).

```{literalinclude} ../../examples/atr/atr_4_evaluate.py
:language: python
```
