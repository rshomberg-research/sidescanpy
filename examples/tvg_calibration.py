"""
Fit the TVG terms a dataset still needs and apply them.

Backscatter is binned by slant range, four loss models (flat,
spreading-only, absorption-only, both) are fitted to the decay, and the
winner's spreading/absorption terms are passed to ``correct()``. The
figure shows the fit and the mean level vs. range before and after.

Usage:
    python examples/tvg_calibration.py survey.ss.h5 --output-dir out/

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import analysis
from sidescanpy.core.units import linear_to_db
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    backscatter_vs_range,
    compute_geometric_ranges,
    correct,
    invert,
    slice_by_pings,
)
from sidescanpy.viz.tvg import plot_mean_vs_range, plot_range_decay_fit

logger = logging.getLogger("tvg_calibration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", help="freq_key, e.g. hz_1495000 (default: the first one)")
    parser.add_argument("--pings", type=int, default=5000, help="Use the first N pings")
    parser.add_argument("--bin-width-m", type=float, default=2.0)
    parser.add_argument("--usable-range-m", type=float, help="Far edge of the fit (default: estimated)")
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, and undo any recorded corrections: the fit needs the signal
    #    as recorded. Slant ranges come from the geometric step.
    survey = read_survey(args.input, chunks={"ping": 512})
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = slice_by_pings(survey.get_frequency(freq_key).dataset, stop=args.pings).load()
    ds = compute_geometric_ranges(invert(ds))

    # 2. Far edge of the fit: where the signal reaches the noise floor.
    usable_range_m = args.usable_range_m
    if usable_range_m is None:
        usable_range_m = analysis.estimate_usable_range(ds, bin_width_m=args.bin_width_m).usable_range_m
    if usable_range_m is None:
        logger.error("could not estimate the usable range; pass --usable-range-m")
        return 1

    # 3. Fit the decay. "auto" places the fit window past the near-field
    #    peak and the reference range a third of the way into it.
    decay = analysis.fit_range_decay(
        ds, bin_width_m=args.bin_width_m, usable_range_m=usable_range_m, r_ref_m="auto", r_fit_min_m="auto",
    )
    if decay.fit is None:
        logger.error("no decay model could be fitted: %s", "; ".join(decay.notes))
        return 1
    best = decay.fit.best
    tvg = {"spreading_loss_db": best.params.spreading_loss_db, "absorption_db_per_m": best.params.absorption_db_per_m}
    logger.info(
        "best model %s (RMSE %.2f dB, tvg_status=%s) -> tvg params %s",
        decay.best_candidate, decay.best_rmse_db, decay.tvg_status, tvg,
    )

    # 4. Apply exactly those terms.
    corrected = correct(ds, ["tvg"], {"tvg": tvg})

    # 5. Before/after: mean level vs. slant range (uncorrected shown in dB).
    before = ds.assign(backscatter_db=linear_to_db(ds["backscatter_intensity"]))
    fig, (ax_fit, ax_before, ax_after) = plt.subplots(1, 3, figsize=(18, 5))
    plot_range_decay_fit(decay, ax=ax_fit)
    plot_mean_vs_range(backscatter_vs_range(before, "backscatter_db", "slant_range_m", args.bin_width_m), ax=ax_before,
                       title="Uncorrected (dB)")
    plot_mean_vs_range(backscatter_vs_range(corrected, "backscatter_intensity", "slant_range_m", args.bin_width_m),
                       ax=ax_after, title=f"After TVG ({decay.best_candidate})")
    for ax in (ax_before, ax_after):
        ax.axvspan(decay.r_fit_min_m, decay.r_fit_max_m, color="0.9", zorder=0, label="fit window")
        ax.set_xlabel("Slant range (m)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tvg_calibration.png", dpi=120)
    plt.close(fig)

    # 6. How flat is the corrected curve inside the fit window?
    after = backscatter_vs_range(corrected, "backscatter_intensity", "slant_range_m", args.bin_width_m)
    centers = ((after["range_min_m"] + after["range_max_m"]) / 2).to_numpy()
    in_window = (centers >= decay.r_fit_min_m) & (centers <= decay.r_fit_max_m) & np.isfinite(after["mean"].to_numpy())
    slope = float(np.polyfit(centers[in_window], after["mean"].to_numpy()[in_window], 1)[0]) if in_window.sum() > 1 else None
    (args.output_dir / "tvg_params.json").write_text(json.dumps({"freq_key": freq_key, "tvg": tvg, "fit": decay.to_dict(),
                                                                 "corrected_slope_db_per_m": slope}, indent=2, default=str))
    logger.info("corrected slope in fit window: %s dB/m -> %s", None if slope is None else f"{slope:.3f}", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
