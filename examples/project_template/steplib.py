"""
Shared plumbing for the template's step scripts: config loading, output
paths, logging, step reports and figure saving.

Project code, not library code: copy it with the step scripts and change
it freely. The processing itself is sidescanpy's API, called directly from
each step.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sidescanpy import __version__ as SIDESCANPY_VERSION
from sidescanpy.log import configure_logging

logger = logging.getLogger("project")


@dataclass
class Paths:
    """Where a survey's inputs are and where every step writes."""

    config_dir: Path
    raw_dir: Path | None
    targets_source_dir: Path | None
    output_root: Path

    def resolve(self, p: str | Path) -> Path:
        """A path from the config, relative to the config file's directory."""
        p = Path(p)
        return p if p.is_absolute() else self.config_dir / p

    @property
    def ss_h5(self) -> Path:
        return self.output_root / "ss_h5"

    @property
    def targets_json(self) -> Path:
        return self.output_root / "targets" / "targets.json"

    @property
    def reports(self) -> Path:
        return self.output_root / "reports"

    def step_dir(self, name: str) -> Path:
        return self.output_root / name


def load_config(path: str | Path) -> tuple[dict, Paths]:
    """Read the YAML config and resolve its paths against its own directory."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = path.parent
    p = cfg.get("paths", {})

    def opt(key):
        return (base / p[key]) if p.get(key) else None

    paths = Paths(config_dir=base, raw_dir=opt("raw_dir"), targets_source_dir=opt("targets_source_dir"),
                  output_root=base / p.get("output_root", "output"))
    cfg["_config_path"] = str(path)
    return cfg, paths


def ss_h5_files(directory: Path) -> list[Path]:
    """Every ``.ss.h5`` in ``directory``, sorted (survey order)."""
    return sorted(directory.glob("*.ss.h5"))


@dataclass
class StepReport:
    """What one step did, written to ``reports/<step>.json`` and ``.md``."""

    step: str
    config_path: str
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    results: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def note(self, text: str) -> None:
        """Something the reader should check before the next step."""
        logger.warning(text)
        self.notes.append(text)

    def figure(self, path: Path, root: Path) -> None:
        self.figures.append(str(path.relative_to(root)))

    def write(self, reports_dir: Path, next_step: str | None) -> Path:
        reports_dir.mkdir(parents=True, exist_ok=True)
        elapsed = time.perf_counter() - self._t0
        data = {
            "step": self.step, "config": self.config_path, "sidescanpy": SIDESCANPY_VERSION,
            "started_utc": self.started_utc, "elapsed_s": round(elapsed, 1),
            "results": self.results, "figures": self.figures, "notes": self.notes, "next": next_step,
        }
        json_path = reports_dir / f"{self.step}.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))
        lines = [f"# {self.step}", "", f"- config: `{self.config_path}`", f"- sidescanpy {SIDESCANPY_VERSION}",
                 f"- started {self.started_utc}, took {elapsed / 60:.1f} min", ""]
        if self.notes:
            lines += ["## Check these", "", *[f"- {n}" for n in self.notes], ""]
        lines += ["## Results", "", "```json", json.dumps(self.results, indent=2, default=str), "```", ""]
        if self.figures:
            lines += ["## QA figures", "", *[f"- `{f}`" for f in self.figures], ""]
        if next_step:
            lines += ["## Next", "", f"`python {next_step} {self.config_path}`", ""]
        (reports_dir / f"{self.step}.md").write_text("\n".join(lines))
        logger.info("%s done in %.1f min; report: %s", self.step, elapsed / 60, reports_dir / f"{self.step}.md")
        return json_path


def start_step(step: str, description: str, argv: list[str] | None = None) -> tuple[dict, Paths, StepReport]:
    """Parse ``<config> [-v]``, log to the console and to
    ``<output_root>/logs/<step>_<timestamp>.log``, and return the config,
    paths and an empty report."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="survey YAML config")
    parser.add_argument("-v", "--verbose", action="store_true", help="also show sidescanpy DEBUG messages")
    args = parser.parse_args(argv)
    cfg, paths = load_config(args.config)
    log_dir = paths.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configure_logging("INFO", log_file=log_dir / f"{step}_{stamp}.log", library_level="DEBUG" if args.verbose else None)
    logger.info("%s -- %s (sidescanpy %s)", step, cfg.get("survey", "?"), SIDESCANPY_VERSION)
    return cfg, paths, StepReport(step=step, config_path=cfg["_config_path"])


def save_figure(fig, path: Path, *, dpi: int = 120, tight: bool = True) -> Path:
    """Save and close a matplotlib figure."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None)
    finally:
        plt.close(fig)
    return path
