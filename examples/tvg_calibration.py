"""
Fit the TVG a dataset still needs, apply it, and check the result.

The median level is binned by slant range and fitted, from past its peak
to the noise knee, with no TVG, absorption only and 20/30/40 log R with
and without absorption (``fit_tvg_models``). The best model's
``spreading_loss_db`` and ``absorption_db_per_m`` go to ``correct()``,
and ``check_tvg`` measures how flat the corrected level is over the same
window. The figure is the fit and the level after TVG.

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

from sidescanpy import analysis
from sidescanpy.io.hdf5 import read_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import compute_geometric_ranges, correct, invert, slice_by_pings
from sidescanpy.viz.tvg import plot_tvg_fit

logger = logging.getLogger("tvg_calibration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A .ss.h5 file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", help="freq_key, e.g. hz_1495000 (default: the first one)")
    parser.add_argument("--pings", type=int, default=5000, help="Use the first N pings")
    parser.add_argument("--bin-width-m", type=float, default=1.0)
    parser.add_argument("--window-m", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Fit window (default: 5 m past the peak to the noise knee)")
    args = parser.parse_args()
    configure_logging("INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, and undo any recorded corrections: the fit needs the signal
    #    as recorded. Slant ranges come from the geometric step.
    survey = read_survey(args.input, chunks={"ping": 512})
    freq_key = args.frequency or sorted(survey.frequencies)[0]
    ds = slice_by_pings(survey.get_frequency(freq_key).dataset, stop=args.pings).load()
    ds = compute_geometric_ranges(invert(ds))

    # 2. Fit the models. The best one is the TVG to apply.
    fit = analysis.fit_tvg_models(ds, bin_width_m=args.bin_width_m, window_m=args.window_m)
    if fit.best is None:
        logger.error("no TVG model could be fitted: %s", "; ".join(fit.notes))
        return 1
    tvg = {"spreading_loss_db": fit.tvg_params.spreading_loss_db,
           "absorption_db_per_m": fit.tvg_params.absorption_db_per_m}
    logger.info("best model %s (%.2f dB RMS over %.1f-%.1f m, free n %.1f) -> tvg params %s",
                fit.best, fit.best_model.rms_db, *fit.window_m, fit.free_n, tvg)

    # 3. Apply exactly those terms.
    corrected = correct(ds, ["tvg"], {"tvg": tvg})

    # 4. A good fit is a hypothesis: apply it and check the level is flat
    #    over the same window.
    check = analysis.check_tvg(ds, fit.tvg_params, fit.window_m, bin_width_m=args.bin_width_m)
    fig, _ = plot_tvg_fit(fit, check)
    fig.savefig(args.output_dir / "tvg_calibration.png", dpi=120)
    plt.close(fig)
    (args.output_dir / "tvg_params.json").write_text(json.dumps(
        {"freq_key": freq_key, "tvg": tvg, "fit": fit.to_dict(), "check": check.to_dict()}, indent=2, default=str,
    ))
    logger.info("after TVG: spread %.2f dB, trend %+.2f dB per 10 m; %d pings corrected -> %s",
                check.spread_db, check.trend_db_per_10m, corrected.sizes["ping"], args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
