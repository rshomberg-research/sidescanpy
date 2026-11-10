# Target recognition

Two common ways of getting from sidescan data to machine-learning target
recognition, step by step. Each step links to an example that does it;
sidescanpy covers the sonar side (reading, correcting, cutting images,
mapping image boxes back to the data), and established tools do the
learning and the annotation.

## From raw data to a two-stage recognizer

Most sidescan target recognition runs in two stages: a **detector**
finds anything on the seabed in whole swath images and may raise false
alarms, then a **classifier** looks at a small chip around each
detection and decides what it is, or rejects it.

| Step | What happens | Example |
|---|---|---|
| 1. Convert and correct | raw JSF, SDF or XTF to the data model, then TVG, ground range and positions, one call per correction | [Getting started](getting_started.md), [Correct a line](examples/plot_correct_a_line.rst) |
| 2. Cut images to annotate | dual-swath tiles with square pixels in metres, plus an index of where each tile sits in the data | [Swath tiles for annotation](examples/plot_swath_tiles.rst) |
| 3. Annotate | draw boxes in FiftyOne, CVAT or Label Studio and export COCO; read the boxes back as targets with ping, side and position | [Swath tiles for annotation](examples/plot_swath_tiles.rst), [Review and correct labels](atr/review.md) |
| 4. Train a detector | one class, "object", on whole images | [Labelled images](atr/prepare.md), [Train a detector](atr/detector.md) |
| 5. Train a classifier | chips around each labelled object, plus clutter and background images | [Train a classifier](atr/classifier.md), [Training dataset with backgrounds](examples/plot_training_dataset.rst) |
| 6. Find candidates | run both stages on new tiles and turn the detections into targets with positions | [Run the models on a survey](atr/survey.md) |
| 7. Review | confirm, reject and correct the candidates; the corrections are the next round's training data | [Review and correct labels](atr/review.md) |
| 8. Test on more data | score on lines or surveys that were never used for training, in image space or in metres | [Test both stages](atr/evaluate.md), [Run the models on a survey](atr/survey.md) |

The [step-by-step pages](atr/index.md) walk through steps 4 to 8 in
turn. The detector and classifier examples train on a public labelled image
set (Santos et al. 2024, mine-like objects seen by a Marine Sonic sonar),
because no openly licensed raw sidescan recording with labelled targets
was available. With your own data, steps 2 and 3 produce the same kind of
images and labels.

## From a target catalog to image chips

When a survey already has picked targets, from acquisition software
(EdgeTech Discover and Klein catalogs are read directly) or from an
analyst, the chips come straight from the catalog.

| Step | What happens | Example or function |
|---|---|---|
| 1. Read the catalog | into `Target` records; keep it as JSON or CSV | `io.read_target_catalog`, `targets.read_targets_json` ([Targets and datasets](datasets.md)) |
| 2. Place targets on pings | by recorded file and ping, by position, or by ping number | `targets.correlate` ([Targets and datasets](datasets.md)) |
| 3. Render chips | at any correction level, pixel size and window style, with COCO and YOLO labels | [Target chips at several corrections and resolutions](examples/plot_target_chips.rst) |
| 4. Add backgrounds and split | images with no target, and a train/validation split that keeps overlapping windows together | [Training dataset with backgrounds](examples/plot_training_dataset.rst) |

## Building blocks

| Function | Does |
|---|---|
| `export.ml.write_swath_tiles` | a corrected channel as PNG swath tiles plus `tiles.json` |
| `export.ml.read_coco_boxes` | boxes from a COCO file (annotations or detector results), per image |
| `export.ml.boxes_to_targets` | boxes on a tile to targets: file, ping, side, sample, position, size in metres, score |
| `export.ml.build_target_dataset` | chips and background images with COCO and YOLO labels, at any correction level, resolution and style |
| `targets.match_targets` | candidates against a catalog within a distance: hits, false alarms, misses, precision, recall |

## Things that go wrong

- **Splitting at random.** Neighbouring swath images overlap and a target
  is often seen on several passes, so a random split puts near-copies in
  training and test and flatters the scores. Split by line, survey or
  date.
- **Backgrounds that differ from the targets' surroundings.** If
  background images come from different ranges or look different after
  the display stretch, a classifier learns that instead of the objects.
  Look at them side by side before training.
- **Another sonar.** A model trained on one sonar, frequency or recorder
  setting rarely transfers to another without new labels. Correcting the
  data first (TVG, ground range, the same pixel size) narrows the gap but
  does not close it.

```{toctree}
:hidden:

Step by step <atr/index>
```
