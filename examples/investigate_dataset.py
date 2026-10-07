"""
Characterize an unfamiliar sidescan dataset: units, nadir, beam geometry,
usable range and TVG status.

A linear walkthrough of the sidescanpy analysis API. Each step calls one
library function, keeps its result, and saves the matching plot. The
results go to ``metrics.json`` for comparison across runs. The
``sidescan-analyze investigate`` command runs the same flow.

Usage:
    python examples/investigate_dataset.py mission.ss.h5 --frequency hz_1495000 --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy import analysis
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    blank_nadir_noise,
    compute_geometric_ranges,
    invert,
    uniform_exclusion_boundary,
)
from sidescanpy.viz import analysis as plots
from sidescanpy.viz.tvg import plot_range_decay_fit, plot_range_decay_residuals

logger = logging.getLogger("investigate_dataset")


def save(fig_and_axes, path: Path) -> None:
    fig, _ = fig_and_axes
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--frequency", help="freq_key, e.g. hz_1495000 (default: the first one)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pings", type=int, default=8000, help="Analyze the first N pings")
    args = parser.parse_args()
    configure_logging("INFO")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load one frequency and prepare it: undo any recorded corrections
    #    (we want the signal as recorded), zero the electronic noise burst
    #    at nadir, and add slant/ground ranges.
    survey = read_survey(args.input)
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = survey.get_frequency(freq_key).dataset.isel(ping=slice(0, args.pings)).load()
    ds = invert(ds)
    ds = blank_nadir_noise(ds)
    ds = compute_geometric_ranges(ds)
    var = "backscatter_intensity"
    logger.info("%s / %s: %d pings", args.input.name, freq_key, ds.sizes["ping"])

    # 2. What do the values look like?
    values = analysis.characterize_values(ds, value_var=var)
    save(plots.plot_value_histograms(ds, var), out / "value_histograms.png")
    save(plots.plot_sample_ping(ds, var), out / "sample_ping.png")

    # 3. How much would range-binned outlier rejection remove?
    outliers = analysis.compare_outlier_rejection(ds, value_var=var, bin_width_m=2.0)
    save(plots.plot_outlier_comparison(ds, outliers, var), out / "outlier_comparison.png")

    # 4. Nadir: gap / bright spike / noisy, and where the bottom return starts.
    nadir = analysis.analyze_nadir(ds, value_var=var)
    save(plots.plot_cross_track_profile(ds, var), out / "cross_track_profile.png")
    save(plots.plot_nadir_zoom(ds, var, nadir=nadir), out / "nadir_zoom.png")
    save(plots.plot_nadir_waterfall(ds, var, nadir=nadir), out / "nadir_waterfall.png")

    # 5. Usable range: where the signal sinks into the noise floor. The
    #    nadir gap is masked per ping first.
    usable = analysis.estimate_usable_range(
        ds, value_var=var, bin_width_m=2.0, exclude_before_sample=nadir.first_bottom_return_sample,
    )
    save(plots.plot_usable_range(usable), out / "usable_range.png")

    # 6. Beam geometry, triangulated from the nadir boundary and usable range.
    beam = analysis.estimate_beam_geometry(ds, nadir.first_bottom_return_sample, usable.usable_range_m)
    save(plots.plot_beam_geometry(ds, beam, var), out / "beam_geometry.png")

    # 7. TVG status: which spreading/absorption terms are still in the data.
    #    One survey-wide nadir boundary here (see uniform_exclusion_boundary).
    boundary = nadir.first_bottom_return_sample
    decay = analysis.fit_range_decay(
        ds, value_var=var, bin_width_m=5.0, usable_range_m=usable.usable_range_m,
        exclude_before_sample=uniform_exclusion_boundary(boundary) if boundary is not None else None,
    )
    save(plot_range_decay_fit(decay), out / "range_decay_fit.png")
    save(plot_range_decay_residuals(decay), out / "range_decay_residuals.png")

    # 8. Units, from the value range cross-checked against the decay fit.
    units = analysis.infer_units(values, decay)
    save(plots.plot_units_distribution(ds, var), out / "units_distribution.png")

    results = {
        "values": values, "outlier_comparison": outliers, "nadir": nadir, "usable_range": usable,
        "beam_geometry": beam, "range_decay": decay, "units": units,
    }
    metrics = {"source": str(args.input), "freq_key": freq_key, **{k: r.to_dict() for k, r in results.items()}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "units=%s nadir=%s usable_range_m=%s tvg_status=%s -> %s",
        units.units, nadir.nadir_type, usable.usable_range_m, decay.tvg_status, out,
    )


if __name__ == "__main__":
    main()
