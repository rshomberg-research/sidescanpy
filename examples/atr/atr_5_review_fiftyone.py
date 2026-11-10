"""
Step 5: review the detections in FiftyOne and correct the labels in CVAT.

Needs fiftyone, a CVAT account and a person; exports the corrected labels as COCO.

Usage: python examples/atr/atr_5_review_fiftyone.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import fiftyone as fo
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos")
CLASSES = ["MILCO", "NOMBO", "clutter"]
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = YOLO(root / "runs" / "detector" / "weights" / "best.pt")
classifier = models.resnet18()
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSES))
classifier.load_state_dict(torch.load(root / "classifier.pt", map_location=device))
classifier = classifier.to(device).eval()
to_tensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def chip(image, xc, yc, w, h):
    W, H = image.size
    side = max(32.0, 3 * max(w * W, h * H))
    x0, y0 = xc * W - side / 2, yc * H - side / 2
    return image.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((64, 64))


# %%
# Load the test images with their labels (``ground_truth``) and the
# two-stage output (``predictions``). FiftyOne boxes are
# ``[x_left, y_top, width, height]`` relative to the image.

dataset = fo.Dataset("santos_2017_review", overwrite=True)
for path in sorted((root / "images" / "2017").glob("*.jpg")):
    image = Image.open(path).convert("RGB")
    t = path.with_suffix(".txt").read_text().split()
    truth = [fo.Detection(label=CLASSES[int(t[i])],
                          bounding_box=[float(t[i + 1]) - float(t[i + 3]) / 2,
                                        float(t[i + 2]) - float(t[i + 4]) / 2,
                                        float(t[i + 3]), float(t[i + 4])])
             for i in range(0, len(t), 5)]
    preds = []
    result = detector.predict(path, conf=0.05, imgsz=640, verbose=False)[0]
    for (xc, yc, w, h), conf in zip(result.boxes.xywhn.tolist(), result.boxes.conf.tolist()):
        with torch.no_grad():
            k = int(classifier(to_tensor(chip(image, xc, yc, w, h))[None].to(device)).argmax())
        if CLASSES[k] != "clutter":
            preds.append(fo.Detection(label=CLASSES[k], bounding_box=[xc - w / 2, yc - h / 2, w, h],
                                      confidence=conf))
    dataset.add_sample(fo.Sample(filepath=str(path), ground_truth=fo.Detections(detections=truth),
                                 predictions=fo.Detections(detections=preds)))

# %%
# Score the predictions against the labels inside FiftyOne, so the app
# can show true positives, false positives and misses per image.

dataset.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval", iou=0.1)
session = fo.launch_app(dataset)

# %%
# Send the images with any false positive or miss to CVAT, correct the
# boxes and classes there, and load the corrections back.

to_fix = dataset.match((fo.ViewField("eval_fp") > 0) | (fo.ViewField("eval_fn") > 0))
to_fix.annotate("review", backend="cvat", label_field="ground_truth",
                classes=["MILCO", "NOMBO"], launch_editor=True)
input("Press Enter when the corrections are saved in CVAT...")
dataset.load_annotations("review")

# %%
# Export the corrected labels as COCO for the next training round, or for
# sidescanpy to turn into targets on the sonar data.

dataset.export(export_dir=str(root / "reviewed_2017"), dataset_type=fo.types.COCODetectionDataset,
               label_field="ground_truth")
