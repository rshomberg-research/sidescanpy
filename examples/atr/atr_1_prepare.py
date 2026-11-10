"""
Step 1: download the labelled sonar images and split them by year.

Downloads Santos et al. (2024), 'Side-scan sonar imaging for mine detection' (CC BY 4.0,
doi:10.6084/m9.figshare.24574879), checks each yearly archive, and writes a one-class
detection dataset in the Ultralytics layout: train 2010/2015/2018, validation 2021,
test 2017.

Usage: python examples/atr/atr_1_prepare.py ATR_DIR

See "Target recognition" in the sidescanpy docs for what each step does and its results.
"""

# 2026 rshomberg@uri.edu

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "atr_santos")
raw = root / "raw"
raw.mkdir(parents=True, exist_ok=True)

# figshare file id and SHA-256 of each yearly archive (version 2 of the record).
ARCHIVES = {
    "2010": (43169008, "e06e952f088320e113f43c33a43ff3c1ed0e3ec4abce045838c5ef76477e4ed3"),
    "2015": (43169002, "526f4a3917609ec793ded9f6632633fbb081944c7f4e506222e4e1eb18dad48b"),
    "2017": (43169005, "17cdf0d5a29dbaaa6dabc24da344e87693b36929280c5a9d72194bdc86527ff8"),
    "2018": (43169011, "e91bce177db051751f9a6c03e8f97c0dbf17c45178cafb414330169ae4108e62"),
    "2021": (43168999, "435c676844d34e35568bcfd68c619f289a1a32fe23b12abcfe1afe37d3756d30"),
}
SPLITS = {"train": ["2010", "2015", "2018"], "val": ["2021"], "test": ["2017"]}
CLASSES = ["MILCO", "NOMBO"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# %%
# Download and check each archive.

for year, (file_id, digest) in ARCHIVES.items():
    path = raw / f"{year}.zip"
    if not path.exists() or sha256(path) != digest:
        urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{file_id}", path)
    if sha256(path) != digest:
        raise OSError(f"{path}: checksum does not match")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(root / "images")

# %%
# One label file per image: ``class x_centre y_centre width height``,
# relative to the image size (images are 416 or 1024 pixels square). The
# detection set keeps the boxes but drops the class; the original labels
# stay in ``images/<year>/`` for the classifier.

counts = {}
for split, years in SPLITS.items():
    img_dir = root / "detect" / "images" / split
    lbl_dir = root / "detect" / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for year in years:
        for jpg in sorted((root / "images" / year).glob("*.jpg")):
            tokens = jpg.with_suffix(".txt").read_text().split()
            boxes = [tokens[i:i + 5] for i in range(0, len(tokens), 5)]
            shutil.copy2(jpg, img_dir / jpg.name)
            (lbl_dir / f"{jpg.stem}.txt").write_text("".join(f"0 {' '.join(b[1:])}\n" for b in boxes))
            n_img += 1
            n_box += len(boxes)
    counts[split] = (n_img, n_box)

(root / "detect" / "data.yaml").write_text(
    f"path: {(root / 'detect').resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: object\n"
)
for split, (n_img, n_box) in counts.items():
    print(f"{split:5s} {n_img:4d} images, {n_box:4d} boxes")
