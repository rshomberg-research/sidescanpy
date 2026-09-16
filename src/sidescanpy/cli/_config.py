"""
Shared ``--config`` support for the ``sidescan-*`` CLIs: a small
per-dataset JSON settings file (e.g. gamma, TVG term overrides, channel
restriction — see the family root's ``DATASET_BRIEF_PLAYBOOK.md``
"settings that genuinely need to vary per dataset" table) that seeds
argparse defaults for a command, so an operator doesn't have to retype
a long, survey-specific flag string by hand every run.

Explicit CLI flags always override a config value — this only changes
argparse *defaults*, via ``set_defaults()`` called before
``parse_args()``; a flag actually typed on the command line still wins,
same as any other argparse default.

JSON, not YAML: this codebase already uses JSON as its one interchange
format everywhere (``targets.json``, ``metrics.json``, ``summary.json``)
and has no existing YAML dependency — this avoids a new dependency for
no real gain over a format already in use throughout.

2026 rshomberg@uri.edu
"""

import argparse
import json
from pathlib import Path


def find_config_path(argv: list[str]) -> Path | None:
    """Pull ``--config``/``--config=PATH`` out of ``argv`` directly,
    without a full parse — the subcommand (and therefore which
    subparser's defaults to seed) isn't known yet at this point."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    """Seed ``parser``'s defaults from ``config``, restricted to keys
    that are actually valid ``dest`` names on it.

    A shared config file may hold settings meant for other
    ``sidescan-*`` (sub)commands too — unrecognized keys are silently
    ignored here, not an error, so one project-wide per-dataset config
    file can be reused across every command without per-command
    filtering by the caller.
    """
    dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in config.items() if k in dests})
