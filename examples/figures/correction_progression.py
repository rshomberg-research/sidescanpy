"""
Figure recipe: the same targets at every correction step, side by side.

One row per target, one column per render level -- ``uncorrected``,
``db`` (20 log10, no TVG), ``radiometric`` (TVG), ``geometric``
(along/across-track meters) and ``georef`` (UTM) -- each cut with the
library's own export extractors
(:func:`~sidescanpy.export.windows.extract_pixel_style_window`,
:func:`~sidescanpy.export.windows.extract_geo_style_window`), so the
figure shows exactly what a dataset export would render. A cross marks
the target. Visual QA of the correction pipeline, not a data product.

``plot_progression`` is the reusable part; ``main`` correlates a target
catalog against corrected ``.ss.h5`` files and saves the figure.

Usage:
    python examples/figures/correction_progression.py corrected/ targets.json --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy.core.target import Target
from sidescanpy.export import to_uint8
from sidescanpy.export.windows import (
    extract_geo_style_window,
    extract_pixel_style_window,
    load_geo_neighborhood,
    mask_nadir_gap_with_uncorrected,
    place_targets_in_file,
    with_db_var,
)
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.targets.correlate import build_position_index, correlate_target
from sidescanpy.targets.io import read_targets_json

logger = logging.getLogger("correction_progression")

STEPS = ("uncorrected", "db", "radiometric", "geometric", "georef")


def progression_windows(ds_full, target: Target, ping_idx: int, *, chip_px: int, geo_px: int, rng) -> dict:
    """``{step: (image, (row, col) or None)}`` for one target."""
    channel = target.channel if target.channel is not None else 0
    items = {channel: [(target, ping_idx)]}
    out = {}

    half = chip_px // 2 + 50
    lo = max(0, ping_idx - half)
    masked = with_db_var(mask_nadir_gap_with_uncorrected(ds_full.isel(ping=slice(lo, ping_idx + half)).load()))
    placed_by_channel, placed_combined = place_targets_in_file(items, ds_full.sizes["sample"])
    for step in STEPS[:3]:
        rendered = extract_pixel_style_window(
            masked, center_sample=target.sample, channel_idx=channel, local_ping=ping_idx - lo, ping_offset=lo,
            style="chip", level=step, size_px=chip_px, jitter_px=0, rng=rng,
            placed_by_channel=placed_by_channel, placed_combined=placed_combined,
        )
        out[step] = _image_and_mark(rendered)

    for step in STEPS[3:]:
        try:
            view = load_geo_neighborhood(
                ds_full, target, ping_idx, items, ["chip"], size_px=geo_px, jitter_px=0, resolution_m=None,
                with_db=False, per_sample_latlon=step == "georef",
            )
        except KeyError as exc:
            logger.warning("target %s: no %s in the file, %s column left blank", target.target_id, exc, step)
            out[step] = (None, None)
            continue
        rendered = extract_geo_style_window(
            view.ds, center_sample=view.target.sample, channel_idx=channel, local_ping=view.local_ping, style="chip",
            level=step, size_px=geo_px, jitter_px=0, rng=rng, mps=view.mps, items_by_channel=view.items, lo=view.lo,
            primary=view.target,
        )
        out[step] = _image_and_mark(rendered)
    return out


def _image_and_mark(rendered):
    if rendered is None or not rendered.targets:
        return (None if rendered is None else rendered.image), None
    t = rendered.targets[0]
    return rendered.image, (float(t.extra_attributes["window_row"]), float(t.sample))


def plot_progression(rows: list[tuple[str, dict]], *, gamma: float = 1.0):
    """Grid figure: one row per ``(label, progression_windows(...))``."""
    fig, axes = plt.subplots(len(rows), len(STEPS), figsize=(2.2 * len(STEPS), 2.2 * len(rows)), squeeze=False)
    for r, (label, windows) in enumerate(rows):
        for c, step in enumerate(STEPS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            image, mark = windows.get(step, (None, None))
            if image is not None and np.isfinite(image).any():
                ax.imshow(to_uint8(image, gamma=gamma)[0], cmap="gray", vmin=0, vmax=255)
                if mark is not None:
                    ax.plot(mark[1], mark[0], "r+", markersize=10)
            else:
                ax.set_facecolor("black")
            if r == 0:
                ax.set_title(step, fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.tight_layout()
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ss_h5_dir", type=Path, help="Directory of corrected .ss.h5 files")
    parser.add_argument("targets_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-targets", type=int, default=7)
    parser.add_argument("--chip-size-px", type=int, default=200)
    parser.add_argument("--geo-size-px", type=int, default=200)
    parser.add_argument("--max-match-dist-m", type=float, default=25.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = build_position_index(args.ss_h5_dir)
    rng = np.random.default_rng(0)
    rows = []
    for target in read_targets_json(args.targets_json):
        if len(rows) >= args.n_targets:
            break
        match = correlate_target(target, index, args.max_match_dist_m)
        if match is None or target.sample is None:
            continue
        ds_full = read_survey(match.path, chunks={"ping": 512}).get_frequency(match.freq_key).dataset
        windows = progression_windows(ds_full, target, match.ping_idx, chip_px=args.chip_size_px, geo_px=args.geo_size_px, rng=rng)
        rows.append((f"{target.target_id}\n{match.path.name}", windows))
    if not rows:
        raise SystemExit("no target correlated to a ping")

    fig, _axes = plot_progression(rows, gamma=args.gamma)
    out = args.output_dir / "correction_progression.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s (%d targets)", out, len(rows))


if __name__ == "__main__":
    main()
