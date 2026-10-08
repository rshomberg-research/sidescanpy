"""
matplotlib is the optional ``viz`` extra: everything outside
``sidescanpy.viz`` (and the CLI's figure modules) must import and run
without it.

Each check runs in a subprocess with ``sys.modules["matplotlib"] = None``,
which makes any ``import matplotlib`` raise ImportError exactly as if it
were not installed.

2026 rshomberg@uri.edu
"""

import subprocess
import sys
import textwrap

import numpy as np

from sidescanpy.core import SideScanSurvey
from sidescanpy.io.hdf5 import write_survey
from tests.unit.synthetic import make_raw_dataset

_BLOCK = "import sys; sys.modules['matplotlib'] = None\n"

# CLI modules that are figure code by design (imported lazily by the
# subcommands that need them).
_NEEDS_MATPLOTLIB = ("sidescanpy.viz", "sidescanpy.cli._figures", "sidescanpy.cli._investigate")


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK + textwrap.dedent(code), *args],
        capture_output=True, text=True, timeout=300,
    )


def test_every_non_viz_module_imports_without_matplotlib():
    proc = _run(f"""
        import importlib, pkgutil
        import sidescanpy
        skip = {_NEEDS_MATPLOTLIB!r}
        failed = []
        names = [m.name for m in pkgutil.walk_packages(sidescanpy.__path__, "sidescanpy.")]
        for name in names:
            if name.startswith(skip):
                continue
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed.append(f"{{name}}: {{exc!r}}")
        for name in ("sidescanpy.cli.convert", "sidescanpy.cli.correct"):
            assert name in names, name
        assert not failed, "\\n".join(failed)
    """)
    assert proc.returncode == 0, proc.stderr


def _synthetic_ss_h5(path):
    survey = SideScanSurvey(source_files=[], source_format="synthetic", survey_name="synthetic")
    data = np.random.default_rng(0).uniform(1.0, 10.0, (10, 2, 50)).astype(np.float32)
    survey.add_frequency("hz_200000", make_raw_dataset(backscatter=data))
    write_survey(survey, path)
    return path


def test_sidescan_correct_runs_without_matplotlib(tmp_path):
    _synthetic_ss_h5(tmp_path / "a.ss.h5")
    proc = _run("""
        from sidescanpy.cli.correct import main
        sys.argv = ["sidescan-correct", "--ss-h5-dir", sys.argv[1], "--tvg"]
        sys.exit(main())
    """, str(tmp_path))
    assert proc.returncode == 0, proc.stderr


def test_figure_subcommand_fails_with_one_line_error(tmp_path):
    proc = _run("""
        from sidescanpy.cli.export import main
        sys.argv = ["sidescan-export", "waterfall", "--ss-h5-dir", sys.argv[1], "--output-dir", sys.argv[1]]
        sys.exit(main())
    """, str(tmp_path))
    assert proc.returncode == 1
    errors = [line for line in proc.stderr.splitlines() if "ERROR" in line]
    assert len(errors) == 1, proc.stderr
    assert "sidescan-export waterfall needs matplotlib" in errors[0]
    assert "sidescanpy[viz]" in errors[0]
    assert "Traceback" not in proc.stderr


def test_analyze_subcommand_fails_with_one_line_error(tmp_path):
    proc = _run("""
        from sidescanpy.cli.analyze import main
        sys.argv = ["sidescan-analyze", "units", "--input", sys.argv[1], "--frequency", "200000", "--output-dir", sys.argv[1]]
        sys.exit(main())
    """, str(tmp_path / "missing.ss.h5"))
    assert proc.returncode == 1
    assert "sidescan-analyze units needs matplotlib" in proc.stderr
    assert "Traceback" not in proc.stderr
