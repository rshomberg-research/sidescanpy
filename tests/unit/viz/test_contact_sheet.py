"""
Smoke tests for sidescanpy.viz.contact_sheet.

Matches how viz/waterfall.py and viz/tvg_diagnostics.py are tested:
figure produced, correct shape, no exceptions — not pixel comparisons.

2026 rshomberg@uri.edu
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sidescanpy.targets.core import Target
from sidescanpy.viz.contact_sheet import contact_sheet, render_target_context


@pytest.fixture
def sample_target() -> Target:
    return Target(
        target_id="0001",
        source_format="jsf_discover",
        catalog_source="Target-0001.target",
        ping=105,
        channel=0,
        sample=15,
        position_confidence="verified",
        extra_attributes={"window_row": 5},
    )


def test_contact_sheet_produces_figure_with_right_grid():
    rng = np.random.default_rng(0)
    images = [rng.random((20, 30)) for _ in range(7)]
    captions = [f"Target {i}" for i in range(7)]
    fig = contact_sheet(images, captions, ncols=4)
    axes = fig.get_axes()
    assert len(axes) == 8  # 2 rows x 4 cols, last cell empty


def test_contact_sheet_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        contact_sheet([np.zeros((5, 5))], ["a", "b"])


def test_contact_sheet_uses_equal_aspect_not_auto():
    # Real bug found 2026-09-17: aspect="auto" stretched genuinely
    # square chip files to fill whatever box shape the grid layout
    # produced for that cell -- confirmed every underlying file was
    # actually square on disk, so this was purely a display bug.
    images = [np.random.default_rng(0).random((20, 20))]
    fig = contact_sheet(images, ["a"])
    ax = fig.get_axes()[0]
    assert ax.get_aspect() == 1.0  # matplotlib reports "equal" as 1.0


def test_contact_sheet_rejects_mismatched_boxes_length(sample_target):
    with pytest.raises(ValueError, match="boxes"):
        contact_sheet([np.zeros((5, 5)), np.zeros((5, 5))], ["a", "b"], boxes=[[sample_target]])


def test_contact_sheet_draws_boxes_when_given(sample_target):
    images = [np.random.default_rng(0).random((20, 20))]
    fig = contact_sheet(images, ["a"], boxes=[[sample_target]], box_size=(6, 6))
    ax = fig.get_axes()[0]
    assert len(ax.patches) == 1


def test_contact_sheet_no_boxes_by_default():
    images = [np.random.default_rng(0).random((20, 20))]
    fig = contact_sheet(images, ["a"])
    ax = fig.get_axes()[0]
    assert len(ax.patches) == 0


def test_contact_sheet_empty_box_list_draws_nothing(sample_target):
    images = [np.random.default_rng(0).random((20, 20)), np.random.default_rng(1).random((20, 20))]
    fig = contact_sheet(images, ["a", "b"], boxes=[[sample_target], []])
    axes = fig.get_axes()
    assert len(axes[0].patches) == 1
    assert len(axes[1].patches) == 0


def test_contact_sheet_multiple_targets_per_thumbnail(sample_target):
    from dataclasses import replace
    second = replace(sample_target, target_id="0002", sample=20)
    images = [np.random.default_rng(0).random((20, 30))]
    fig = contact_sheet(images, ["a"], boxes=[[sample_target, second]], box_size=(6, 6))
    ax = fig.get_axes()[0]
    assert len(ax.patches) == 2


def test_render_target_context_draws_box(sample_target):
    rng = np.random.default_rng(0)
    window = rng.random((10, 30))
    fig = render_target_context(window, sample_target, box_size=(6, 6))
    ax = fig.get_axes()[0]
    patches = ax.patches
    assert len(patches) == 1
    assert "0001" in ax.get_title()


def test_render_target_context_requires_computed_position():
    t = Target(target_id="0002", source_format="jsf_discover", catalog_source="x")
    window = np.zeros((10, 10))
    with pytest.raises(ValueError):
        render_target_context(window, t)


def test_render_target_context_defaults_to_pixel_labels(sample_target):
    window = np.zeros((10, 30))
    fig = render_target_context(window, sample_target, box_size=(6, 6))
    ax = fig.get_axes()[0]
    assert ax.get_xlabel() == "Sample"
    assert ax.get_ylabel() == "Ping (window-local)"


def test_render_target_context_geometric_labels_along_across_track(sample_target):
    window = np.zeros((10, 30))
    fig = render_target_context(window, sample_target, box_size=(6, 6), coordinate_system="geometric")
    ax = fig.get_axes()[0]
    assert "across-track" in ax.get_xlabel().lower()
    assert "along-track" in ax.get_ylabel().lower()


def test_render_target_context_georef_labels_easting_northing(sample_target):
    window = np.zeros((10, 30))
    fig = render_target_context(window, sample_target, box_size=(6, 6), coordinate_system="georef")
    ax = fig.get_axes()[0]
    assert "easting" in ax.get_xlabel().lower()
    assert "northing" in ax.get_ylabel().lower()
