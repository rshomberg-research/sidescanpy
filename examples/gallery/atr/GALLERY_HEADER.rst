Automatic target recognition
----------------------------

Two stages, as in most sidescan target recognition: a detector that finds
anything on the seabed (and is allowed false alarms), then a classifier
that looks closely at each detection. These pages train both on a public
labelled image set, test them on a year they never saw, show how a person
reviews and corrects the results, and run them on sidescanpy tiles with
positions. :doc:`/atr` shows how they connect to the sonar-data steps
above.

They need extra packages (Ultralytics YOLO, torchvision, FiftyOne) and a
GPU, so they are not run when this site is built: the code is shown with
the output and figures of a run on a workstation with one 20 GB GPU.
