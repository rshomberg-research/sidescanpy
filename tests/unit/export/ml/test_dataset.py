"""
Unit tests for sidescanpy.export.ml.dataset: DatasetSpec and build_target_dataset.

Most cases render a small corrected synthetic survey end to end.

2026 rshomberg@uri.edu
"""

import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sidescanpy.core.target import Target
from sidescanpy.export.ml import DatasetSpec, build_target_dataset
from sidescanpy.targets.io import read_targets_json, write_targets_json

geo_windows = importlib.import_module("sidescanpy.export.windows.geo")
neighborhood = importlib.import_module("sidescanpy.export.windows.neighborhood")


@dataclass
class _Case:
    ss_h5_dir: Path
    targets_json: Path
    output: Path
    spec: DatasetSpec


def _case(*, targets_json, ss_h5_dir, output, **spec_fields) -> _Case:
    """A build request in the shape the CLI-era tests used (flat keyword overrides)."""
    if isinstance(spec_fields.get("background_ping_ranges"), str):
        spec_fields["background_ping_ranges"] = json.loads(spec_fields["background_ping_ranges"])
    return _Case(ss_h5_dir, targets_json, output, DatasetSpec(**spec_fields))


def _run(case: _Case) -> int:
    """0 when any image was written, else 1."""
    result = build_target_dataset(case.ss_h5_dir, read_targets_json(case.targets_json), case.spec, case.output)
    return 0 if result.n_images else 1


def _swath_target(tid="7", ping=150, channel=1, sample=15, confidence="verified", n_pings=300):
    return Target(
        target_id=tid, source_format="x", catalog_source="x",
        latitude=float(np.linspace(40.0000, 40.0010, n_pings)[ping]),
        longitude=float(np.linspace(-70.0000, -70.0010, n_pings)[ping]),
        channel=channel, sample=sample, position_confidence=confidence,
    )


class TestDatasetSpec:
    def test_defaults_match_the_cli(self):
        spec = DatasetSpec()
        assert spec.styles == ("chip", "single-swath", "dual-swath")
        assert (spec.size_px, spec.swath_size_px, spec.render_level) == (400, 512, "radiometric")
        assert spec.n_background == 0 and spec.val_fraction == 0.0 and spec.layout == "flat"

    def test_round_trip_through_dict_and_yaml(self, tmp_path):
        spec = DatasetSpec(
            styles=["chip"], render_level="geometric", resolution_m=[0.05, 0.1], categories=["a", "b"],
            n_background=3, background_ping_ranges={"f.ss.h5": [[0, 10]]}, val_fraction=0.2, layout="ultralytics",
        )
        assert DatasetSpec.from_dict(spec.to_dict()) == spec
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(spec.to_dict()))
        assert DatasetSpec.from_yaml(path) == spec
        assert json.loads(json.dumps(spec.to_dict())) == spec.to_dict()

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="sizepx"):
            DatasetSpec.from_dict({"sizepx": 10})

    @pytest.mark.parametrize("bad", [
        {"styles": ["tile"]}, {"render_level": "raw"}, {"bit_depth": 12}, {"layout": "coco"},
        {"val_fraction": 1.0}, {"normalization_scope": "global"},
    ])
    def test_invalid_values_rejected(self, bad):
        with pytest.raises(ValueError):
            DatasetSpec(**bad)

    def test_normalizes_inputs(self):
        spec = DatasetSpec(styles=["chip"], resolution_m=[0.2], n_augment=None, n_background=None)
        assert spec.styles == ("chip",) and spec.resolution_m == 0.2
        assert spec.n_augment == 1 and spec.n_background == 0


class TestBuildTargetDataset:
    def _build(self, tmp_path, ss_h5_dir, targets, out="out", **spec):
        spec = {"styles": ["chip"], "size_px": 16, "swath_size_px": 16, "seed": 0, **spec}
        return build_target_dataset(ss_h5_dir, targets, DatasetSpec(**spec), tmp_path / out)

    def test_result_summary_and_manifest(self, tmp_path, swath_ss_h5_dir):
        targets = [_swath_target(), _swath_target("far", confidence="verified"), _swath_target("low", confidence="unknown")]
        targets[1] = replace(targets[1], latitude=1.0, longitude=1.0)
        result = self._build(tmp_path, swath_ss_h5_dir, targets)
        assert (result.n_targets, result.n_matched) == (2, 1)
        assert {(s.target_id, s.reason.split()[0]) for s in result.skipped} == {("far", "no"), ("low", "below")}
        chip = result.styles["chip"]
        assert chip.images == [tmp_path / "out" / "chip" / "target_7.png"]
        assert (chip.n_target_images, chip.n_background_images, chip.n_annotations) == (1, 0, 1)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["outputs"]["chip"]["images"] == 1
        assert DatasetSpec.from_dict(manifest["spec"]).styles == ("chip",)
        assert not (tmp_path / "out" / "chip" / "contact_sheet.png").exists()

    def test_accepts_explicit_file_list(self, tmp_path, swath_ss_h5_dir):
        result = self._build(tmp_path, sorted(swath_ss_h5_dir.glob("*.ss.h5")), [_swath_target()])
        assert result.n_images == 1

    def test_nothing_matched_writes_nothing(self, tmp_path, swath_ss_h5_dir):
        result = self._build(tmp_path, swath_ss_h5_dir, [replace(_swath_target(), latitude=1.0, longitude=1.0)])
        assert result.n_matched == 0 and result.styles == {} and result.manifest_path is None
        assert not (tmp_path / "out" / "chip").exists()

    def test_unknown_category_raises_before_rendering(self, tmp_path, swath_ss_h5_dir):
        with pytest.raises(ValueError, match="class_z"):
            self._build(tmp_path, swath_ss_h5_dir, [replace(_swath_target(), category="class_z")], categories=["class_a"])

    def test_hooks_receive_sheet_and_context(self, tmp_path, swath_ss_h5_dir):
        sheets, contexts = [], []
        spec = DatasetSpec(styles=["chip"], size_px=16, with_context=True, seed=0)
        build_target_dataset(
            swath_ss_h5_dir, [_swath_target()], spec, tmp_path / "out",
            contact_sheet=lambda thumbs, captions, **kw: sheets.append((len(thumbs), captions, kw)),
            context=lambda image, target, **kw: contexts.append((image.shape, target.target_id, kw)),
        )
        ((n, captions, kw),) = sheets
        assert n == 1 and captions[0].startswith("7 ") and kw["path"].name == "contact_sheet.png"
        ((_shape, tid, ckw),) = contexts
        assert tid == "7" and ckw["coordinate_system"] == "pixel" and ckw["path"].name == "context_7.png"

    def test_several_resolutions_get_their_own_directories(self, tmp_path, swath_ss_h5_dir):
        result = self._build(tmp_path, swath_ss_h5_dir, [_swath_target()], render_level="geometric", resolution_m=[0.002, 0.004])
        assert sorted(result.styles) == ["res_0.002m/chip", "res_0.004m/chip"]
        for key in result.styles:
            assert (tmp_path / "out" / key / "target_7.png").exists()
            assert (tmp_path / "out" / key / "targets.coco.json").exists()

    def test_split_by_target_id_hash(self, tmp_path, swath_ss_h5_dir):
        targets = [_swath_target(str(i), ping=p) for i, p in enumerate((60, 120, 180, 240))]
        result = self._build(tmp_path, swath_ss_h5_dir, targets, val_fraction=0.5, n_augment=2, jitter_px=1)
        chip = tmp_path / "out" / "chip"
        train = (chip / "splits" / "train.txt").read_text().split()
        val = (chip / "splits" / "val.txt").read_text().split()
        assert sorted(train + val) == sorted(p.name for p in result.styles["chip"].images)
        for tid in map(str, range(4)):
            # Both draws of a target land in the same split.
            assert {f"target_{tid}_a0.png" in val, f"target_{tid}_a1.png" in val} in ({True}, {False})
        assert train and val
        log = json.loads((chip / "augmentations.json").read_text())
        assert {e["split"] for e in log} == {"train", "val"}
        rerun = self._build(tmp_path, swath_ss_h5_dir, targets, out="again", val_fraction=0.5, n_augment=2, jitter_px=1)
        assert rerun.styles["chip"].split_counts == result.styles["chip"].split_counts

    def test_ultralytics_layout(self, tmp_path, swath_ss_h5_dir):
        targets = [_swath_target(str(i), ping=p) for i, p in enumerate((60, 120, 180, 240))]
        result = self._build(
            tmp_path, swath_ss_h5_dir, targets, layout="ultralytics", val_fraction=0.5,
            n_background=1, background_max_nan_frac=1.0,
        )
        chip = tmp_path / "out" / "chip"
        data = yaml.safe_load((chip / "data.yaml").read_text())
        assert data["train"] == "images/train" and data["names"] == {0: "target"}
        for split in ("train", "val"):
            if not (chip / "images" / split).is_dir():
                continue
            for png in (chip / "images" / split).glob("*.png"):
                assert (chip / "labels" / split / f"{png.stem}.txt").exists()
        bg = [p for p in result.styles["chip"].images if p.name.startswith("bg_")]
        assert len(bg) == 1
        split = bg[0].parent.name
        assert (chip / "labels" / split / "bg_0.txt").read_text() == ""
        assert not (chip / "targets.yolo").exists()
        n_images = sum(len(json.loads(p.read_text())["images"]) for p in chip.glob("*.coco.json"))
        assert n_images == len(result.styles["chip"].images)


class TestDatasetBuild:
    """Render levels, value variables, resolution, bit depth and annotation boxes."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="radiometric",
            size_px=200, swath_size_px=200, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return _case(**defaults)

    def _matched_target(self):
        lat = float(np.linspace(40.0000, 40.0010, 20)[10])
        lon = float(np.linspace(-70.0000, -70.0010, 20)[10])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=500,
            position_confidence="verified",
        )

    def test_value_var_raw_changes_geometric_chip_pixels(self, tmp_path, georef_ss_h5_dir):
        """--value-var is the direct answer to the other half of the same
        2026-09-28 request: raw vs. radiometric independently selectable
        from the geometric/georef coordinate choice, not fixed to
        radiometric. A real pixel-value difference, not just "doesn't
        crash" -- backscatter_raw vs. backscatter_intensity are genuinely
        different arrays (TVG correction changes the values)."""
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        out_radiometric = tmp_path / "out_radiometric"
        args_radiometric = self._args(
            tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
            value_var="radiometric", output=out_radiometric, seed=0,
        )
        assert _run(args_radiometric) == 0

        out_raw = tmp_path / "out_raw"
        args_raw = self._args(
            tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
            value_var="uncorrected", output=out_raw, seed=0,
        )
        assert _run(args_raw) == 0

        img_radiometric = np.asarray(Image.open(out_radiometric / "chip" / "target_7.png"))
        img_raw = np.asarray(Image.open(out_raw / "chip" / "target_7.png"))
        assert img_radiometric.shape == img_raw.shape
        assert not np.array_equal(img_radiometric, img_raw)

    def test_value_var_db_changes_geometric_chip_pixels(self, tmp_path, georef_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        imgs = {}
        for value_var in ("uncorrected", "db"):
            out = tmp_path / f"out_{value_var}"
            args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric", value_var=value_var, output=out, seed=0)
            assert _run(args) == 0
            imgs[value_var] = np.asarray(Image.open(out / "chip" / "target_7.png"))
        assert imgs["db"].shape == imgs["uncorrected"].shape
        assert not np.array_equal(imgs["db"], imgs["uncorrected"])

    def test_value_var_uncorrected_has_no_effect_and_warns_under_radiometric_render_level(self, tmp_path, georef_ss_h5_dir, caplog):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._matched_target()], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", value_var="uncorrected")
        assert _run(args) == 0
        assert any("value_var=uncorrected has no effect" in r.message for r in caplog.records)

    def test_min_confidence_reaches_coco_yolo_writers(self, tmp_path, georef_ss_h5_dir):
        """Real bug found running this against real Klein SDF data: Klein
        targets are unconditionally position_confidence='unknown', and
        write_coco()/write_yolo() have their OWN independent
        min_confidence='verified' default, separate from run_chip's own
        target-list filter above -- passing --min-confidence unknown
        got real crops written for every target but zero COCO/YOLO
        annotations for any of them, because run_chip never threaded
        args.min_confidence through to either writer call."""
        target = replace(self._matched_target(), position_confidence="unknown")
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, min_confidence="unknown")
        assert _run(args) == 0

        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1

    def _swath_matched_target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    def test_geometric_render_level_calls_geometric_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        # extract_geometric_window_griddata (scipy, the DEFAULT
        # resample_method) -- 2026-09-29, real user direction after
        # reviewing a real-data comparison of kdtree/binned/griddata:
        # griddata chosen as the default, all three kept selectable via
        # --resample-method (see cli/export.py's own comment at the call
        # site and TestRunChipResampleMethod below for the dispatch
        # coverage across all three options).
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric",
            styles=["chip", "single-swath", "dual-swath"], swath_size_px=64,
        )
        assert _run(args) == 0
        assert spy.call_count == 3  # once per style
        for style in ("chip", "single-swath", "dual-swath"):
            assert (tmp_path / "out" / style / "target_7.png").exists()

    def test_resolution_m_override_reaches_geometric_extraction_for_every_style(self, tmp_path, swath_ss_h5_dir, mocker):
        """Real production report: some targets' chip/single-swath/dual-swath
        came out 'way zoomed out', mostly-black with the real swath tiny in
        the center -- traced to the per-target auto-calculated resolution_m
        (derived from a wide ping-neighborhood-average spacing) blowing up
        for a target sitting in a locally slow/dense-ping region. --resolution-m
        lets the operator set a known-good value directly, bypassing that
        auto-calculation entirely for every style."""
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric",
            styles=["chip", "single-swath", "dual-swath"], swath_size_px=64,
            resolution_m=0.25,
        )
        assert _run(args) == 0
        assert spy.call_count == 3
        for call in spy.call_args_list:
            assert call.kwargs["resolution_m"] == 0.25


    def test_resolution_m_override_reaches_georef_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_georef_window")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="georef",
            styles=["chip"], swath_size_px=64, resolution_m=0.1,
        )
        assert _run(args) == 0
        assert spy.call_count == 1
        assert spy.call_args.kwargs["resolution_m"] == 0.1

    def test_resolution_m_override_ignored_with_a_warning_for_pixel_styles(self, tmp_path, swath_ss_h5_dir, caplog):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric",
            styles=["chip"], resolution_m=0.25,
        )
        with caplog.at_level("WARNING"):
            assert _run(args) == 0
        assert any("has no effect" in r.message for r in caplog.records)

    def test_georef_render_level_calls_georef_extraction(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_georef_window")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, render_level="georef",
            styles=["chip", "dual-swath"], swath_size_px=64,
        )
        assert _run(args) == 0
        assert spy.call_count == 2
        for style in ("chip", "dual-swath"):
            assert (tmp_path / "out" / style / "target_7.png").exists()

    def test_geometric_render_level_annotates_co_located_targets(self, tmp_path, swath_ss_h5_dir):
        """Real-data gap: _extract_geo_style_window used to only ever
        annotate the primary target, unlike the raw-pixel styles' own
        targets_within_window search -- two targets close enough to share
        one geometric chip must both end up in that chip's own COCO
        annotations, not just the one the crop was centered on."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        primary = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[150]), longitude=float(lon[150]), channel=1, sample=15,
            position_confidence="verified",
        )
        co_located = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[152]), longitude=float(lon[152]), channel=1, sample=16,
            position_confidence="verified",
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([primary, co_located], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"])
        assert _run(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        file_names = {img["file_name"] for img in coco["images"]}
        # each target still gets its own centered chip file...
        assert {"target_7.png", "target_8.png"} <= file_names
        # ...but target 8's chip should also carry an annotation for target 7
        # (found via co-target search), not just its own -- 2+ annotations
        # total for at least one of the two images.
        by_image = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], 0)
            by_image[ann["image_id"]] += 1
        assert max(by_image.values()) >= 2

    def test_raw_and_radiometric_render_levels_produce_different_images(self, tmp_path, swath_ss_h5_dir):
        """backscatter_raw and backscatter_intensity are byte-identical in
        this fixture only if TVG is a no-op there -- correct_survey
        with 'tvg' genuinely modifies backscatter_intensity, so raw and
        radiometric renders of the same target must differ."""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        raw_args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="uncorrected", styles=["chip"], output=tmp_path / "raw_out")
        radiometric_args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=tmp_path / "rad_out")
        assert _run(raw_args) == 0
        assert _run(radiometric_args) == 0

        raw_img = np.array(Image.open(tmp_path / "raw_out" / "chip" / "target_7.png"))
        rad_img = np.array(Image.open(tmp_path / "rad_out" / "chip" / "target_7.png"))
        assert not np.array_equal(raw_img, rad_img)

    def test_bit_depth_16_writes_16bit_png_with_same_annotations(self, tmp_path, swath_ss_h5_dir):
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        out8, out16 = tmp_path / "out8", tmp_path / "out16"
        assert _run(self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=out8)) == 0
        assert _run(self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], output=out16, bit_depth=16)) == 0

        img8 = np.array(Image.open(out8 / "chip" / "target_7.png"))
        img16 = np.array(Image.open(out16 / "chip" / "target_7.png"))
        assert img8.dtype == np.uint8
        assert img16.dtype == np.uint16
        assert img16.max() > 255
        assert np.abs((img16 >> 8).astype(int) - img8.astype(int)).max() <= 1
        coco8 = json.loads((out8 / "chip" / "targets.coco.json").read_text())
        coco16 = json.loads((out16 / "chip" / "targets.coco.json").read_text())
        assert coco8["annotations"] == coco16["annotations"]

    def test_db_render_level_differs_from_raw(self, tmp_path, swath_ss_h5_dir):
        """'db' is 20*log10(backscatter_raw) with no TVG -- a nonlinear
        transform of raw, so its percentile-stretched 8-bit render differs
        from raw's. (Not compared against radiometric: this fixture's whole
        swath is ~0.06m, all inside TVG's min_range_m clamp, so its gain is
        a constant offset that percentile stretching removes.)"""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._swath_matched_target()], targets_json)

        imgs = {}
        for level in ("uncorrected", "db"):
            args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level=level, styles=["chip"], output=tmp_path / f"{level}_out")
            assert _run(args) == 0
            imgs[level] = np.array(Image.open(tmp_path / f"{level}_out" / "chip" / "target_7.png"))
        assert not np.array_equal(imgs["db"], imgs["uncorrected"])

    def test_pixel_style_uses_real_bbox_size(self, tmp_path, georef_ss_h5_dir):
        """Regression: chip/COCO output had every target's box
        at the same size, regardless of each target's own real, measured
        extent (e.g. a catalog's 4-corner columns). "chip"
        style's native window IS size_px already (extract_chip produces
        that size directly, no separate resize step) -- size_px=8 here,
        well inside georef_ss_h5_dir's 20 real pings, so no resize is
        triggered on either axis and the expected bbox comes through
        unscaled, an exact check."""
        t = replace(self._matched_target(), bbox=(490.0, 95.0, 4.0, 3.0))
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], size_px=8)
        assert _run(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        ann = coco["annotations"][0]
        assert ann["bbox"][2] == pytest.approx(4.0)
        assert ann["bbox"][3] == pytest.approx(3.0)

    def test_pixel_style_no_bbox_falls_back_to_shared_box_size(self, tmp_path, georef_ss_h5_dir):
        """A center-only pick (bbox=None, e.g. JSF/EdgeTech Discover's own
        catalog) must still fall back to the shared --box-size placeholder
        exactly as before this fix existed -- not a regression for the
        format this project was originally built around."""
        t = self._matched_target()
        assert t.bbox is None
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)

        args = self._args(tmp_path, georef_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], box_size=40.0)
        assert _run(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        ann = coco["annotations"][0]
        assert ann["bbox"][2] == pytest.approx(40.0)
        assert ann["bbox"][3] == pytest.approx(40.0)

    def test_pixel_style_different_targets_get_different_bbox_sizes(self, tmp_path, swath_ss_h5_dir):
        """The actual "all bounding boxes are the same size" symptom,
        reproduced directly: two targets with genuinely different
        measured extents must come out with genuinely different COCO box
        sizes, not the one shared placeholder for both. size_px=10 here,
        well inside swath_ss_h5_dir's 30 real samples (and 300 real
        pings), so no resize is triggered on either axis."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        small = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[100]), longitude=float(lon[100]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 145.0, 3.0, 2.0),
        )
        big = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[200]), longitude=float(lon[200]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 245.0, 9.0, 7.0),
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([small, big], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="radiometric", styles=["chip"], size_px=10)
        assert _run(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        sizes = {
            next(img["file_name"] for img in coco["images"] if img["id"] == ann["image_id"]): tuple(ann["bbox"][2:])
            for ann in coco["annotations"]
        }
        assert sizes["target_7.png"] != sizes["target_8.png"]
        assert sizes["target_7.png"] == pytest.approx((3.0, 2.0))
        assert sizes["target_8.png"] == pytest.approx((9.0, 7.0))

    def test_geometric_style_different_targets_get_different_bbox_sizes(self, tmp_path, swath_ss_h5_dir):
        """Same real bug, geo-rendered path -- this is the render level
        used for geometric export (--render-level geometric). Exact pixel values
        depend on real meters-per-sample/meters-per-ping/resolution_m
        conversion, so this checks the sizes differ and are neither the
        shared placeholder, not exact numbers."""
        n_pings = 300
        lat = np.linspace(40.0000, 40.0010, n_pings)
        lon = np.linspace(-70.0000, -70.0010, n_pings)
        small = Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=float(lat[100]), longitude=float(lon[100]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 145.0, 4.0, 4.0),
        )
        big = Target(
            target_id="8", source_format="x", catalog_source="x",
            latitude=float(lat[200]), longitude=float(lon[200]), channel=1, sample=15,
            position_confidence="verified", bbox=(9.0, 245.0, 40.0, 40.0),
        )
        targets_json = tmp_path / "targets.json"
        write_targets_json([small, big], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"], box_size=99.0)
        assert _run(args) == 0

        coco = json.loads((tmp_path / "out" / "chip" / "targets.coco.json").read_text())
        sizes = {
            next(img["file_name"] for img in coco["images"] if img["id"] == ann["image_id"]): tuple(ann["bbox"][2:])
            for ann in coco["annotations"]
        }
        # Not asserting *which* one comes out bigger -- resolution_m
        # (meters per output pixel) is recomputed independently per
        # target/window from that window's own local geometry, so it
        # isn't necessarily comparable across two different ping
        # positions the way it would be for two crops of the same
        # window. The real bug this test guards is "every target gets
        # the same box regardless of its own measured extent" -- these
        # two assertions are the direct, robust check for that.
        assert sizes["target_7.png"] != sizes["target_8.png"]
        # neither equals the shared placeholder (chosen far from either
        # target's real extent to avoid a coincidental collision) --
        # confirms both actually used their own measured extent, not a
        # fallback.
        assert sizes["target_7.png"] != (99.0, 99.0)
        assert sizes["target_8.png"] != (99.0, 99.0)

    def test_geometric_level_out_of_bounds_sample_skips_gracefully(self, tmp_path, swath_ss_h5_dir):
        """Real crash found running this against real Klein SDF data: a
        target whose catalog entry carries no ping_meters_per_sample
        (match_frequency then picks a frequency arbitrarily) can be
        correlated to a channel whose own sample count is smaller than
        the sample index the target's position was originally captured
        against elsewhere -- an exact .isel(sample=t.sample) then raised
        an uncaught IndexError instead of being skipped like every other
        unavailable-window case."""
        target = replace(self._swath_matched_target(), sample=999)  # fixture's n_samples=30
        targets_json = tmp_path / "targets.json"
        write_targets_json([target], targets_json)

        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, render_level="geometric", styles=["chip"])
        assert _run(args) == 1  # gracefully -- not a crash

class TestResampleMethod:
    """2026-09-29: --resample-method selects which of the three
    extract_geometric_window* functions the 'geometric' render level
    uses, default 'griddata' -- real user direction after reviewing a
    real-data comparison of all three."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return _case(**defaults)

    def _target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    @pytest.mark.parametrize(
        "resample_method,spied_fn",
        [
            ("kdtree", "extract_geometric_window"),
            ("binned", "extract_geometric_window_binned"),
            ("griddata", "extract_geometric_window_griddata"),
        ],
    )
    def test_each_option_calls_its_own_function(self, tmp_path, swath_ss_h5_dir, mocker, resample_method, spied_fn):
        spy = mocker.spy(geo_windows, spied_fn)
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, resample_method=resample_method)
        assert _run(args) == 0
        assert spy.call_count == 1

    def test_omitting_resample_method_defaults_to_griddata(self, tmp_path, swath_ss_h5_dir, mocker):
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        # No resample_method key at all -- matches an argparse.Namespace
        # built before this flag existed, or a caller that never sets it.
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=swath_ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=40.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        args = _case(**defaults)
        assert _run(args) == 0
        assert spy.call_count == 1

    def test_all_three_options_produce_square_output(self, tmp_path, swath_ss_h5_dir):
        # Real, user-confirmed requirement: no resample_method should
        # ever produce non-square "geometric" render-level output.
        for resample_method in ("kdtree", "binned", "griddata"):
            targets_json = tmp_path / f"targets_{resample_method}.json"
            write_targets_json([self._target()], targets_json)
            args = self._args(
                tmp_path, swath_ss_h5_dir, targets_json, resample_method=resample_method,
                output=tmp_path / f"out_{resample_method}",
            )
            assert _run(args) == 0
            from PIL import Image
            img = Image.open(tmp_path / f"out_{resample_method}" / "chip" / "target_7.png")
            assert img.size == (64, 64)

class TestSwathPreResampleDownsample:
    """Block-averaging the neighborhood before resampling must not move annotations."""

    def test_annotation_matches_full_resolution_resample(self, tmp_path, georef_ss_h5_dir, monkeypatch):
        targets_json = tmp_path / "targets.json"
        write_targets_json([TestDatasetBuild()._matched_target()], targets_json)

        def run(out, oversample):
            monkeypatch.setattr(neighborhood, "SWATH_OVERSAMPLE", oversample)
            args = TestDatasetBuild()._args(
                tmp_path, georef_ss_h5_dir, targets_json, render_level="geometric",
                styles=["single-swath", "dual-swath"], swath_size_px=64, output=out,
            )
            assert _run(args) == 0
            return {
                style: json.loads((out / style / "targets.coco.json").read_text())["annotations"][0]["bbox"]
                for style in ("single-swath", "dual-swath")
            }

        averaged = run(tmp_path / "averaged", 2.0)
        full = run(tmp_path / "full", 1e9)  # every factor clamps to 1: no averaging
        for style in averaged:
            np.testing.assert_allclose(averaged[style][:2], full[style][:2], atol=2.0)

class TestAugmentation:
    """--n-augment / --random-flip / --across-axis slant / --box-size-m."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip"], render_level="geometric",
            size_px=64, swath_size_px=64, box_size=10.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None,
        )
        defaults.update(overrides)
        return _case(**defaults)

    def _target(self, n_pings=300):
        lat = float(np.linspace(40.0000, 40.0010, n_pings)[150])
        lon = float(np.linspace(-70.0000, -70.0010, n_pings)[150])
        return Target(
            target_id="7", source_format="x", catalog_source="x",
            latitude=lat, longitude=lon, channel=1, sample=15,
            position_confidence="verified",
        )

    def _yolo_centers(self, out_dir, style="chip"):
        labels = sorted((out_dir / style / "targets.yolo" / "labels").glob("*.txt"))
        return {p.stem: [tuple(map(float, line.split()[1:3])) for line in p.read_text().splitlines()] for p in labels}

    def test_n_augment_writes_one_image_and_label_per_draw(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, n_augment=4, jitter_px=20)
        assert _run(args) == 0

        chip_dir = tmp_path / "out" / "chip"
        assert sorted(p.name for p in chip_dir.glob("target_*.png")) == [f"target_7_a{k}.png" for k in range(4)]
        centers = self._yolo_centers(tmp_path / "out")
        assert set(centers) == {f"target_7_a{k}" for k in range(4)}
        # independent jitter draws -> target lands in different places
        assert len({c[0] for c in centers.values()}) > 1
        log = json.loads((chip_dir / "augmentations.json").read_text())
        assert [e["augmentation"] for e in log] == [0, 1, 2, 3]

    def test_n_augment_default_keeps_single_file_naming(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        assert _run(self._args(tmp_path, swath_ss_h5_dir, targets_json)) == 0
        assert (tmp_path / "out" / "chip" / "target_7.png").exists()
        assert not (tmp_path / "out" / "chip" / "augmentations.json").exists()

    def test_random_flip_mirrors_image_and_label_together(self, tmp_path, swath_ss_h5_dir):
        """Same seed, same jitter draw: the flipped run's image must be the
        unflipped image mirrored, and its label center mirrored to match."""
        from PIL import Image

        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        plain = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "plain", n_augment=6)
        flipped = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "flip", n_augment=6, random_flip=True)
        assert _run(plain) == 0
        assert _run(flipped) == 0

        log = json.loads((tmp_path / "flip" / "chip" / "augmentations.json").read_text())
        assert any(e["flip_h"] or e["flip_v"] for e in log)
        plain_c, flip_c = self._yolo_centers(tmp_path / "plain"), self._yolo_centers(tmp_path / "flip")
        for e in log:
            stem = Path(e["file_name"]).stem
            a = np.array(Image.open(tmp_path / "plain" / "chip" / e["file_name"]))
            b = np.array(Image.open(tmp_path / "flip" / "chip" / e["file_name"]))
            if e["flip_h"]:
                a = a[:, ::-1]
            if e["flip_v"]:
                a = a[::-1, :]
            np.testing.assert_array_equal(a, b)
            (cx, cy), (fx, fy) = plain_c[stem][0], flip_c[stem][0]
            assert fx == pytest.approx(1 - cx if e["flip_h"] else cx, abs=1e-5)
            assert fy == pytest.approx(1 - cy if e["flip_v"] else cy, abs=1e-5)

    def test_slant_across_axis_runs_end_to_end(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, across_axis="slant", resolution_m=0.002, n_augment=2, random_flip=True)
        assert _run(args) == 0
        assert len(list((tmp_path / "out" / "chip").glob("target_7_a*.png"))) == 2

    def test_box_size_m_scales_with_resolution(self, tmp_path, swath_ss_h5_dir):
        targets_json = tmp_path / "targets.json"
        write_targets_json([self._target()], targets_json)
        widths = {}
        for res in (0.05, 0.1):
            out = tmp_path / f"r{res}"
            args = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=out, resolution_m=res, box_size_m=1.0)
            assert _run(args) == 0
            line = (out / "chip" / "targets.yolo" / "labels" / "target_7.txt").read_text().split()
            widths[res] = float(line[3]) * 64
        assert widths[0.05] == pytest.approx(20.0, abs=0.5)
        assert widths[0.1] == pytest.approx(10.0, abs=0.5)

class TestSwathFill:
    """--swath-fill end to end (unit tests: tests/unit/export/test_windows_geo.py)."""

    def test_fill_resolution_is_not_floored_at_ping_spacing(self, tmp_path, swath_ss_h5_dir, mocker):
        """Fixture swath (~0.1 m) is far narrower than ping spacing
        (~0.37 m): the old frame floors resolution at ping spacing (swath
        a sliver of the image); --swath-fill uses width / size_px."""
        lat = float(np.linspace(40.0000, 40.0010, 300)[150])
        lon = float(np.linspace(-70.0000, -70.0010, 300)[150])
        t = Target(target_id="7", source_format="x", catalog_source="x", latitude=lat, longitude=lon,
                   channel=1, sample=15, position_confidence="verified")
        targets_json = tmp_path / "targets.json"
        write_targets_json([t], targets_json)
        spy = mocker.spy(geo_windows, "extract_geometric_window_griddata")
        res = {}
        for fill in (False, True):
            spy.reset_mock()
            args = _case(
                targets_json=targets_json, ss_h5_dir=swath_ss_h5_dir, max_match_dist_m=25.0,
                output=tmp_path / f"out{fill}", styles=["dual-swath"], render_level="geometric",
                size_px=64, swath_size_px=64, box_size=10.0, with_context=False, jitter_px=0, gamma=1.0,
                pct_low=1.0, pct_high=99.0, normalization_scope="image", min_confidence="verified", seed=0,
                resolution_m=None, swath_fill=fill,
            )
            assert _run(args) == 0
            res[fill] = spy.call_args.kwargs["resolution_m"]
        assert res[False] > 0.3          # ping-spacing floor
        assert res[True] < 0.01          # ~0.1 m swath / 64 px

class TestBackgrounds:
    """n_background: no-target images rendered through the target path."""

    def _args(self, tmp_path, ss_h5_dir, targets_json, **overrides):
        defaults = dict(
            targets_json=targets_json, ss_h5_dir=ss_h5_dir, max_match_dist_m=25.0,
            output=tmp_path / "out", styles=["chip", "dual-swath"], render_level="geometric",
            size_px=16, swath_size_px=16, box_size=4.0, with_context=False,
            jitter_px=0, gamma=1.0, pct_low=1.0, pct_high=99.0,
            normalization_scope="image", min_confidence="verified", seed=0,
            resolution_m=None, background_max_nan_frac=1.0,
        )
        defaults.update(overrides)
        return _case(**defaults)

    def _write_targets(self, tmp_path):
        targets_json = tmp_path / "targets.json"
        write_targets_json([TestAugmentation()._target()], targets_json)
        return targets_json

    @pytest.mark.parametrize("render_level", ["radiometric", "geometric"])
    def test_background_images_have_empty_labels(self, tmp_path, swath_ss_h5_dir, render_level):
        targets_json = self._write_targets(tmp_path)
        args = self._args(tmp_path, swath_ss_h5_dir, targets_json, n_background=2, render_level=render_level)
        assert _run(args) == 0
        for style in ("chip", "dual-swath"):
            style_dir = tmp_path / "out" / style
            assert sorted(p.name for p in style_dir.glob("bg_*.png")) == ["bg_0.png", "bg_1.png"]
            labels = style_dir / "targets.yolo" / "labels"
            assert (labels / "bg_0.txt").read_text() == "" and (labels / "bg_1.txt").read_text() == ""
            assert (labels / "target_7.txt").read_text().strip()
            coco = json.loads((style_dir / "targets.coco.json").read_text())
            bg_ids = {im["id"] for im in coco["images"] if im["file_name"].startswith("bg_")}
            assert len(bg_ids) == 2
            assert not any(a["image_id"] in bg_ids for a in coco["annotations"])
            log = json.loads((style_dir / "augmentations.json").read_text())
            bg_log = [e for e in log if e["target_id"] is None]
            assert [e["background"] for e in bg_log] == [0, 1]
            assert all(e["n_targets"] == 0 and abs(e["ping"] - 150) > 10 for e in bg_log)

    def test_targets_unchanged_by_backgrounds(self, tmp_path, swath_ss_h5_dir):
        targets_json = self._write_targets(tmp_path)
        plain = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "plain", jitter_px=3)
        with_bg = self._args(tmp_path, swath_ss_h5_dir, targets_json, output=tmp_path / "bg", jitter_px=3, n_background=1)
        assert _run(plain) == 0
        assert _run(with_bg) == 0
        for style in ("chip", "dual-swath"):
            a = (tmp_path / "plain" / style / "target_7.png").read_bytes()
            assert a == (tmp_path / "bg" / style / "target_7.png").read_bytes()
        assert not list((tmp_path / "plain" / "chip").glob("bg_*"))

    def test_augmented_background_naming_and_ping_ranges(self, tmp_path, swath_ss_h5_dir):
        targets_json = self._write_targets(tmp_path)
        args = self._args(
            tmp_path, swath_ss_h5_dir, targets_json, styles=["chip"], n_background=1, n_augment=2, random_flip=True,
            jitter_px=2, background_ping_ranges='{"survey_a.ss.h5": [[0, 60]]}',
        )
        assert _run(args) == 0
        chip_dir = tmp_path / "out" / "chip"
        assert sorted(p.name for p in chip_dir.glob("bg_*.png")) == ["bg_0_a0.png", "bg_0_a1.png"]
        log = [e for e in json.loads((chip_dir / "augmentations.json").read_text()) if e["target_id"] is None]
        assert all(0 <= e["ping"] < 60 for e in log)
        assert {"flip_h", "flip_v", "nan_fraction", "source_file", "freq_key"} <= set(log[0])
