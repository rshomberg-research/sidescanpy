"""
2. Train an object detector
===========================

The first stage finds *something* on the seabed. It is trained on the
one-class set from :ref:`sphx_glr_examples_atr_atr_1_prepare.py` with
`Ultralytics YOLO <https://docs.ultralytics.com/>`_, starting from
weights pretrained on everyday photographs. Ultralytics is not a
sidescanpy dependency; install it next to it (``pip install ultralytics``,
which brings PyTorch).

The detector should miss little: anything it misses is lost, while false
alarms are the classifier's job. So it is judged mainly on recall, and
run later with a low confidence threshold.

Not run when the docs are built (it needs a GPU; about 10 minutes on one
20 GB workstation card). Usage::

    python examples/gallery/atr/atr_2_train_detector.py ATR_DIR
"""

# 2026 rshomberg@uri.edu

import os
import sys
from pathlib import Path

from ultralytics import YOLO

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos").resolve()
runs = root / "runs"
runs.mkdir(parents=True, exist_ok=True)
os.chdir(runs)  # Ultralytics downloads weights into the working directory

# %%
# Train. Sonar images are not flipped up-down by default in YOLO, and
# shouldn't be (a shadow falls away from the sonar); left-right flips
# mirror port and starboard, which is fine for these one-sided images.

model = YOLO("yolo11s.pt")
model.train(
    data=str(root / "detect" / "data.yaml"),
    imgsz=640,
    epochs=80,
    batch=16,
    patience=30,
    flipud=0.0,
    fliplr=0.5,
    project=str(runs),
    name="detector",
    exist_ok=True,
    seed=0,
    workers=4,
    plots=True,
)

# %%
# Score the best weights on the held-out year.

best = YOLO(runs / "detector" / "weights" / "best.pt")
metrics = best.val(data=str(root / "detect" / "data.yaml"), split="test", imgsz=640,
                   conf=0.05, project=str(runs), name="detector_test", exist_ok=True)
p, r, map50, map50_95 = metrics.box.mean_results()
print(f"test (2017): precision {p:.2f}, recall {r:.2f}, mAP50 {map50:.2f}, mAP50-95 {map50_95:.2f}")

# %%
# Result of a run (55 epochs, 6 minutes; early stopping kept epoch 25)::
#
#     test (2017): precision 0.39, recall 0.57, mAP50 0.33, mAP50-95 0.15
#
# .. image:: /assets/atr/detector_training.png
#    :alt: YOLO training and validation losses, precision, recall and mAP per epoch
#
# Modest, from 589 training boxes. Training at the images' full 1024 px
# did worse here (mAP50 0.02 on the test year): the validation year is
# small and early stopping picked an epoch that did not generalize. More
# labelled data, from more surveys, helps more than any setting.
