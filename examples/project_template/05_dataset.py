"""
Step 5 -- COCO/YOLO training dataset from the target catalog, using the
config's `dataset:` block (sidescanpy DatasetSpec fields).

Writes:
  <output_root>/dataset/<style>/          images, targets.coco.json, YOLO labels, contact_sheet.png
  <output_root>/dataset/context/          per-target context view with the true box
  <output_root>/dataset/manifest.json
  <output_root>/reports/05_dataset.{json,md}

Usage:
    python 05_dataset.py config.yaml

2026 rshomberg@uri.edu
"""

import matplotlib

matplotlib.use("Agg")

from steplib import logger, save_figure, start_step

from sidescanpy.export.ml import DatasetSpec, build_target_dataset
from sidescanpy.targets import read_targets_json
from sidescanpy.viz import contact_sheet, render_target_context


def save_contact_sheet(thumbnails, captions, *, boxes, box_size, suptitle, path):
    fig = contact_sheet(thumbnails, captions, boxes=boxes, box_size=box_size, ncols=6, suptitle=suptitle)
    save_figure(fig, path, dpi=150, tight=False)


def save_context(image, target, *, coordinate_system, resolution_m, box_size, path):
    fig = render_target_context(image, target, box_size=box_size, coordinate_system=coordinate_system,
                                resolution_m=resolution_m)
    save_figure(fig, path, dpi=150, tight=False)


def main(argv=None) -> None:
    cfg, paths, report = start_step("05_dataset", __doc__.split("\n\n")[0], argv)
    if not paths.targets_json.exists():
        raise SystemExit(f"no {paths.targets_json} -- configure targets_source_dir and run 01_convert.py")
    spec = DatasetSpec.from_dict(cfg.get("dataset") or {})
    targets = read_targets_json(paths.targets_json)
    result = build_target_dataset(paths.ss_h5, targets, spec, paths.step_dir("dataset"),
                                  contact_sheet=save_contact_sheet, context=save_context)
    report.results.update({
        "n_targets": result.n_targets, "n_matched": result.n_matched, "n_images": result.n_images,
        "styles": {k: {"target_images": s.n_target_images, "background_images": s.n_background_images,
                       "annotations": s.n_annotations} for k, s in result.styles.items()},
        "manifest": str(result.manifest_path.relative_to(paths.output_root)),
    })
    by_reason: dict[str, list[str]] = {}
    for skipped in result.skipped:
        by_reason.setdefault(f"{skipped.reason} ({skipped.style or 'all styles'})", []).append(skipped.target_id)
    for reason, ids in by_reason.items():
        report.note(f"{len(ids)} target(s) skipped, {reason}: {', '.join(ids)}")
    for key in result.styles:
        sheet = paths.step_dir("dataset") / key / "contact_sheet.png"
        if sheet.exists():
            report.figure(sheet, paths.output_root)
    logger.info("%d/%d target(s) matched, %d image(s)", result.n_matched, result.n_targets, result.n_images)
    report.write(paths.reports, next_step=None)


if __name__ == "__main__":
    main()
