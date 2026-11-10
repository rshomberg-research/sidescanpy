"""
Step 6: run both stages on sidescanpy swath tiles and score in meters.

Tiles a corrected public Klein line, turns detections into targets with positions
(boxes_to_targets) and scores them against a catalog (match_targets).

Usage: python examples/atr/atr_6_apply_to_survey.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

from sidescanpy import datasets
from sidescanpy.core.target import Target
from sidescanpy.export.ml import TileBox, boxes_to_targets, write_swath_tiles
from sidescanpy.io import read_and_convert
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
)
from sidescanpy.targets import match_targets, write_targets_json

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos")
CLASSES = ["MILCO", "NOMBO", "clutter"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Convert, correct and tile the line (as in
# examples/gallery/plot_swath_tiles.py, including the merge window
# for this file's two-entry GPS updates).

survey = read_and_convert([datasets.fetch("usgs_grand_bay_150602205800.xtf")], fmt="xtf")
survey = interpolate_aux_onto_pings(survey, merge_window_s=0.3)
survey = apply_tvg(survey)
survey = compute_geometric_ranges(survey)
survey = compute_georeferencing(survey)
ds = survey.get_frequency("hz_100000").dataset
tiles = write_swath_tiles(ds, root / "survey_tiles", ss_h5="grand_bay_150602205800",
                          freq_key="hz_100000", tile_m=30.0, px_m=0.1, overlap_m=5.0,
                          pct_low=5, pct_high=99.7)

# %%
# Detect on each tile, keep what the classifier does not call clutter, and
# turn the boxes into targets. The detector was trained on colour images,
# so the grey tiles are passed as three equal channels.

detector = YOLO(root / "runs" / "detector" / "weights" / "best.pt")
classifier = models.resnet18()
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSES))
classifier.load_state_dict(torch.load(root / "classifier.pt", map_location=device))
classifier = classifier.to(device).eval()
to_tensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

candidates = []
for record in tiles:
    image = Image.open(root / "survey_tiles" / record.file_name).convert("RGB")
    result = detector.predict(image, conf=0.05, imgsz=640, verbose=False)[0]
    boxes = []
    for (x, y, w, h), conf in zip(result.boxes.xywh.tolist(), result.boxes.conf.tolist()):
        side = max(32.0, 3 * max(w, h))
        crop = image.crop((round(x - side / 2), round(y - side / 2),
                           round(x + side / 2), round(y + side / 2))).resize((64, 64))
        with torch.no_grad():
            k = int(classifier(to_tensor(crop)[None].to(device)).argmax())
        if CLASSES[k] != "clutter":
            boxes.append(TileBox(x - w / 2, y - h / 2, w, h, category=CLASSES[k], score=conf))
    candidates += boxes_to_targets(ds, record, boxes, source_format="detector",
                                   catalog_source=str(root / "runs" / "detector"))
write_targets_json(candidates, root / "survey_candidates.json")
print(f"{len(candidates)} candidate targets on {len(tiles)} tiles")

# %%
# Score against the catalog: a candidate within 3 m of a catalog target
# is a hit. ``min_score`` shows how the trade-off moves with the detector
# threshold.

catalog = [
    Target(tid, "manual", "picked by eye", latitude=lat, longitude=lon, category=cat,
           position_confidence="verified")
    for tid, cat, lat, lon in [
        ("c1", "contact", 30.3642369, -88.3602696), ("c2", "contact", 30.3640208, -88.3602611),
        ("c3", "contact", 30.3638172, -88.3604494), ("r1", "reef", 30.3632802, -88.3595788),
        ("r2", "reef", 30.3628830, -88.3599460),
    ]
]
for min_score in (0.05, 0.25, 0.5):
    print(f"min_score {min_score}:", match_targets(candidates, catalog, max_distance_m=3.0,
                                                   min_score=min_score).summary())
