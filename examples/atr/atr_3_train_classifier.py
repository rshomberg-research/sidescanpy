"""
Step 3: train the second-stage chip classifier (torchvision ResNet-18).

MILCO / NOMBO / clutter, with the detector's own false alarms as clutter (hard
negatives). Run after step 2. A few minutes on a GPU.

Usage: python examples/atr/atr_3_train_classifier.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
"""

# 2026 rshomberg@uri.edu

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos")
CLASSES = ["MILCO", "NOMBO", "clutter"]
CHIP_PX = 64
TRAIN_YEARS, VAL_YEARS = ["2010", "2015", "2018"], ["2021"]
rng = random.Random(0)
torch.manual_seed(0)


def read_boxes(txt: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO label file -> (class, x_centre, y_centre, width, height), relative units."""
    t = txt.read_text().split()
    return [(int(t[i]), *map(float, t[i + 1:i + 5])) for i in range(0, len(t), 5)]


def chip(image: Image.Image, xc: float, yc: float, w: float, h: float) -> Image.Image:
    """Square crop, three times the box's longer side (at least 32 px), centred on it."""
    W, H = image.size
    side = max(32.0, 3 * max(w * W, h * H))
    x0, y0 = xc * W - side / 2, yc * H - side / 2
    return image.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((CHIP_PX, CHIP_PX))


def iou(a, b):
    """Boxes as (xc, yc, w, h), relative units."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# %%
# Chips. Each labelled box gives one chip of its class. The detector runs
# on every image at the low threshold used later; each detection that
# overlaps no label gives a clutter chip.

detector = YOLO(root / "runs" / "detector" / "weights" / "best.pt")


def make_chips(years):
    chips, labels = [], []
    for year in years:
        for jpg in sorted((root / "images" / year).glob("*.jpg")):
            image = Image.open(jpg).convert("RGB")
            boxes = read_boxes(jpg.with_suffix(".txt"))
            for c, xc, yc, w, h in boxes:
                chips.append(chip(image, xc, yc, w, h))
                labels.append(c)
            result = detector.predict(jpg, conf=0.05, imgsz=640, verbose=False)[0]
            for det in result.boxes.xywhn.tolist():
                if all(iou(det, b[1:]) < 0.1 for b in boxes):
                    chips.append(chip(image, *det))
                    labels.append(2)
    return chips, labels


train_chips, train_labels = make_chips(TRAIN_YEARS)
val_chips, val_labels = make_chips(VAL_YEARS)
for name, labels in [("train", train_labels), ("val", val_labels)]:
    print(name, {c: labels.count(k) for k, c in enumerate(CLASSES)})

# %%
# Fine-tune. Classes are unbalanced (NOMBO is rare), so the loss weights
# each class by its inverse frequency.
# Augmentation flips left-right only and jitters brightness, as for the
# detector.

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.3, 0.3),
                               transforms.ToTensor(), normalize])
eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)
counts = np.bincount(train_labels, minlength=len(CLASSES))
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def evaluate(chips, labels):
    model.eval()
    with torch.no_grad():
        x = torch.stack([eval_tf(c) for c in chips]).to(device)
        pred = model(x).argmax(1).cpu().numpy()
    return pred, float((pred == np.array(labels)).mean())


best_acc, n = 0.0, len(train_chips)
for _ in range(30):  # epochs
    model.train()
    order = list(range(n))
    rng.shuffle(order)
    for i in range(0, n, 64):
        idx = order[i:i + 64]
        x = torch.stack([train_tf(train_chips[j]) for j in idx]).to(device)
        y = torch.tensor([train_labels[j] for j in idx]).to(device)
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
    _, acc = evaluate(val_chips, val_labels)
    if acc >= best_acc:
        best_acc = acc
        torch.save(model.state_dict(), root / "classifier.pt")
print(f"best validation accuracy {best_acc:.2f}")
