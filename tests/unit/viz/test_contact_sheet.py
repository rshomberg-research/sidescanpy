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
