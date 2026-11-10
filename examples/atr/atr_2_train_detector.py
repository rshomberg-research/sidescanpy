"""
Step 2: train the first-stage object detector (Ultralytics YOLO).

Needs ultralytics (and PyTorch) and a GPU; about 10 minutes on one 20 GB card.

Usage: python examples/atr/atr_2_train_detector.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
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
