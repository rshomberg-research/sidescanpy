"""
6. Run the models on a survey and score in metres
=================================================

The trained stages applied to sidescanpy output: swath tiles cut from a
corrected line, detections turned into targets with positions
(:func:`~sidescanpy.export.ml.boxes_to_targets`), classified, and scored
against a catalog in metres (:func:`~sidescanpy.targets.match_targets`).

The line is the public USGS Grand Bay Klein line and the catalog the
five illustrative picks of :ref:`sphx_glr_examples_plot_target_chips.py`.
The models were trained on a different sonar (Marine Sonic, 900 kHz)
looking for mine-like objects, so their calls here mean little: this
page shows how the pieces connect, not how well these models transfer.
Train on tiles from your own sonar for that.

Not run when the docs are built. Usage::

    python examples/gallery/atr/atr_6_apply_to_survey.py ATR_DIR
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
# :ref:`sphx_glr_examples_plot_swath_tiles.py`).

survey = read_and_convert([datasets.fetch("usgs_grand_bay_150602205800.xtf")], fmt="xtf")
survey = interpolate_aux_onto_pings(survey)
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

# %%
# Output of a run::
#
#     2 candidate targets on 4 tiles
#     min_score 0.05: 0 hit(s), 2 false alarm(s), 5 miss(es) within 3 m: precision 0.00, recall 0.00
#     min_score 0.25: 0 hit(s), 0 false alarm(s), 5 miss(es) within 3 m: precision nan, recall 0.00
#     min_score 0.5: 0 hit(s), 0 false alarm(s), 5 miss(es) within 3 m: precision nan, recall 0.00
#
# As expected for models trained on another sonar, nothing transfers. The
# candidates are still complete targets (``survey_candidates.json``: file,
# ping, side, sample, position, score), ready for review and for
# :func:`~sidescanpy.export.ml.build_target_dataset`.
