"""
Shared ``--config`` support for the ``sidescan-*`` CLIs: a small
per-dataset settings file that seeds a command's argparse defaults, so an
operator does not retype a long, survey-specific flag string every run.

Explicit flags always win: config values only become argparse defaults
(``set_defaults``) before the final parse.

Format: YAML (``.yaml``/``.yml``) for human-edited configs; anything else
is read as JSON, so existing JSON configs keep working.

Keys are option ``dest`` names (``--snr-threshold`` -> ``snr_threshold``).
One file may hold keys for several subcommands; a key that matches no
subcommand's option at all is a WARNING (``--strict-config``: an error).

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from sidescanpy.cli._common import (
    deferred_warnings,
    discard_pending_warnings,
    pending_warnings_mark,
    warn_deferred,
)

logger = logging.getLogger(__name__)

# Never seeded from a config file.
_RESERVED_DESTS = frozenset({"help", "config"})


class CLIParser(argparse.ArgumentParser):
    """``ArgumentParser`` that records each option's ``dest`` in :attr:`option_dests`.

    Subparsers created with ``add_subparsers().add_parser`` are of this class
    too. Options added through mutually exclusive or argument groups are
    recorded as well.
    """

    def __init__(self, *args, **kwargs):
        self.option_dests: set[str] = set()
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        self.option_dests.add(action.dest)
        return action

    def _recording(self, group):
        add = group.add_argument

        def add_argument(*args, **kwargs):
            action = add(*args, **kwargs)
            self.option_dests.add(action.dest)
            return action

        group.add_argument = add_argument
        return group

    def add_argument_group(self, *args, **kwargs):
        return self._recording(super().add_argument_group(*args, **kwargs))

    def add_mutually_exclusive_group(self, **kwargs):
        return self._recording(super().add_mutually_exclusive_group(**kwargs))


def load_config(path: Path) -> dict:
    """Load a ``--config`` file: ``.yaml``/``.yml`` via ``yaml.safe_load``,
    anything else (including ``.json``) via ``json.loads``."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    return json.loads(text)


def parse_args(
    parser: CLIParser,
    argv: Sequence[str],
    *,
    subcommands: "Mapping[str, CLIParser] | None" = None,
) -> argparse.Namespace:
    """Parse ``argv``, seeding defaults from ``--config`` when one is given.

    Parses once to learn the subcommand, ``--config`` and ``--strict-config``;
    with a config, seeds the chosen (sub)parser's defaults from it and parses
    again, so explicit flags still win. ``subcommands`` maps every subcommand
    name (aliases included -- e.g. a subparsers action's ``choices``) to its
    parser; the namespace must then carry the name as ``subcommand``.

    Keys matching no option of any subcommand are logged as a WARNING
    naming the subcommand; with ``--strict-config`` they are a parser error.
    Warnings raised while parsing are queued until
    :func:`~sidescanpy.cli._common.setup_logging_from_args`.
    """
    with deferred_warnings():
        return _parse_args(parser, argv, subcommands)


def _parse_args(parser, argv, subcommands) -> argparse.Namespace:
    mark = pending_warnings_mark()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    # The second parse re-raises any warnings from the first.
    discard_pending_warnings(mark)

    config = load_config(Path(config_path))
    if subcommands:
        name = args.subcommand
        target = subcommands[name]
        known = set(parser.option_dests).union(*(p.option_dests for p in subcommands.values()))
    else:
        name, target, known = parser.prog, parser, parser.option_dests

    unknown = sorted(k for k in config if k not in known or k in _RESERVED_DESTS)
    if unknown:
        if getattr(args, "strict_config", False):
            target.error(f"--strict-config: unknown config key(s) for {name!r}: {', '.join(unknown)}")
        warn_deferred(logger, "config keys not recognized by %r (ignored): %s", name, ", ".join(unknown))
    target.set_defaults(**{k: v for k, v in config.items() if k in target.option_dests and k not in _RESERVED_DESTS})
    return parser.parse_args(argv)
