"""
Step 4: test both stages on the held-out year.

Compares the detector alone with detector plus classifier on 2017, matching boxes by
overlap (IoU >= 0.1), and draws a figure of test images.

Usage: python examples/atr/atr_4_evaluate.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos")
CLASSES = ["MILCO", "NOMBO", "clutter"]
CHIP_PX, IOU_HIT = 64, 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = YOLO(root / "runs" / "detector" / "weights" / "best.pt")
classifier = models.resnet18()
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSES))
classifier.load_state_dict(torch.load(root / "classifier.pt", map_location=device))
classifier = classifier.to(device).eval()
to_tensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def read_boxes(txt):
    t = txt.read_text().split()
    return [(int(t[i]), *map(float, t[i + 1:i + 5])) for i in range(0, len(t), 5)]


def chip(image, xc, yc, w, h):
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


def score(dets, truth):
    """Greedy one-to-one matching per image; returns hits, false alarms, misses."""
    hits = fa = misses = 0
    for name, boxes in truth.items():
        d = sorted(dets.get(name, []), key=lambda r: -r[1])
        used = set()
        for box, _, _ in d:
            j = max(range(len(boxes)), key=lambda k: iou(box, boxes[k][1:]), default=None)
            if j is not None and j not in used and iou(box, boxes[j][1:]) >= IOU_HIT:
                used.add(j)
                hits += 1
            else:
                fa += 1
        misses += len(boxes) - len(used)
    return hits, fa, misses


# %%
# Run both stages on every test image.

images = sorted((root / "images" / "2017").glob("*.jpg"))
truth = {p.name: read_boxes(p.with_suffix(".txt")) for p in images}
alone, alone_low, two_stage = {}, {}, {}
for path in images:
    image = Image.open(path).convert("RGB")
    result = detector.predict(path, conf=0.05, imgsz=640, verbose=False)[0]
    boxes = result.boxes.xywhn.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    alone_low[path.name] = [(tuple(b), float(c), "object") for b, c in zip(boxes, confs)]
    alone[path.name] = [d for d in alone_low[path.name] if d[1] >= 0.25]
    if len(boxes):
        x = torch.stack([to_tensor(chip(image, *b)) for b in boxes]).to(device)
        with torch.no_grad():
            labels = classifier(x).argmax(1).cpu().numpy()
        two_stage[path.name] = [(tuple(b), float(c), CLASSES[k])
                                for b, c, k in zip(boxes, confs, labels) if CLASSES[k] != "clutter"]

# %%
# Results.

for name, dets in [("detector alone, conf 0.25", alone), ("detector alone, conf 0.05", alone_low),
                   ("detector (conf 0.05) + classifier", two_stage)]:
    h, fa, m = score(dets, truth)
    print(f"{name:34s} {h:3d} hits, {fa:3d} false alarms, {m:3d} misses: "
          f"precision {h / max(h + fa, 1):.2f}, recall {h / max(h + m, 1):.2f}")

milco = {n: [b for b in bs if b[0] == 0] for n, bs in truth.items()}
called = {n: [d for d in ds if d[2] == "MILCO"] for n, ds in two_stage.items()}
h, fa, m = score(called, milco)
print(f"{'two-stage, MILCO only':34s} {h:3d} hits, {fa:3d} false alarms, {m:3d} misses")

# %%
# Some test images with their labels (green) and the two-stage output
# (red: MILCO, cyan: NOMBO).

shown = [p for p in images if truth[p.name]][:6]
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
for ax, path in zip(axes.ravel(), shown):
    image = Image.open(path)
    W, H = image.size
    ax.imshow(image)
    for _, xc, yc, w, h in truth[path.name]:
        ax.add_patch(plt.Rectangle(((xc - w / 2) * W - 4, (yc - h / 2) * H - 4), w * W + 8, h * H + 8,
                                   fill=False, edgecolor="lime", linewidth=1.5))
    for (xc, yc, w, h), _, label in two_stage.get(path.name, []):
        ax.add_patch(plt.Rectangle(((xc - w / 2) * W, (yc - h / 2) * H), w * W, h * H, fill=False,
                                   edgecolor="red" if label == "MILCO" else "cyan", linewidth=1.5))
    ax.set_title(path.name, fontsize=9)
    ax.set_axis_off()
(root / "figures").mkdir(exist_ok=True)
fig.savefig(root / "figures" / "atr_test.png", dpi=150)
