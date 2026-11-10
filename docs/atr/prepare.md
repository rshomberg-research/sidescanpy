# 1. Labelled sonar images

The detector and classifier train on Santos et al. (2024), *Side-scan
sonar imaging for mine detection*: 1170 images from a Marine Sonic sonar
on a Gavia AUV, with YOLO boxes for mine-like contacts (`MILCO`, 437)
and non-mine-like bottom objects (`NOMBO`, 231). CC BY 4.0,
[doi:10.6084/m9.figshare.24574879](https://doi.org/10.6084/m9.figshare.24574879);
article [doi:10.1016/j.dib.2024.110132](https://doi.org/10.1016/j.dib.2024.110132).

The images are already waterfalls; no raw sonar file comes with them.
With your own data, the same kind of images and labels come from
[swath tiles](../examples/plot_swath_tiles.rst) and an annotation tool.

The script downloads the five yearly archives (218 MB), checks their
SHA-256, and splits by year. Neighbouring images overlap and show the
same objects, so a random split would put near-copies in both training
and test:

| Split | Years | Images | Boxes |
|---|---|---|---|
| train | 2010, 2015, 2018 | 1029 | 589 |
| validation | 2021 | 48 | 49 |
| test (data the models never see) | 2017 | 93 | 30 |

It writes a one-class detection set, where every box is `object`, in the
folder layout Ultralytics reads; the classifier decides what each object
is. The label files are plain YOLO text, `class x_centre y_centre width
height` relative to the image (416 or 1024 pixels square).

```{literalinclude} ../../examples/atr/atr_1_prepare.py
:language: python
```
