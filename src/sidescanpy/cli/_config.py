"""
Shared ``--config`` support for the ``sidescan-*`` CLIs: a small
per-dataset settings file (e.g. gamma, TVG term overrides, channel
restriction — see the family root's ``DATASET_BRIEF_PLAYBOOK.md``
"settings that genuinely need to vary per dataset" table) that seeds
argparse defaults for a command, so an operator doesn't have to retype
a long, survey-specific flag string by hand every run.

Explicit CLI flags always override a config value — this only changes
argparse *defaults*, via ``set_defaults()`` called before
``parse_args()``; a flag actually typed on the command line still wins,
same as any other argparse default.

**Format convention (superseded 2026-09-18 -- see ``CLAUDE.md``'s
"Consumer-Project Architecture Contract" section): YAML for anything a
human reads or edits, JSON reserved for machine-generated/machine-only
artifacts.** A ``--config`` file is exactly the human-edited case, so
``.yaml``/``.yml`` is now the recommended extension for new configs.
``.json`` is still accepted, unchanged, for every existing config
already written that way -- this is additive, not a breaking format
change.

2026 rshomberg@uri.edu
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from sidescanpy.cli._common import warn_deferred

logger = logging.getLogger(__name__)


def has_strict_flag(argv: list[str]) -> bool:
    """True if ``--strict-config`` appears in ``argv`` (pre-parse, like
    :func:`find_config_path`)."""
    return "--strict-config" in argv


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


def load_config(path: Path) -> dict:
    """Load a ``--config`` file, dispatched on extension: ``.yaml``/
    ``.yml`` via ``yaml.safe_load``, anything else (including ``.json``)
    via ``json.loads`` -- matches every config file already in use
    across the family before YAML support existed, so an un-suffixed or
    oddly-suffixed existing file keeps working exactly as before."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    config: dict,
    *,
    strict: bool = False,
    subcommand: str | None = None,
    check: bool = True,
    known_extra: "set[str] | frozenset[str]" = frozenset(),
) -> None:
    """Seed ``parser``'s defaults from ``config``, restricted to keys
    that are actually valid ``dest`` names on it.

    A shared config file may hold settings meant for other
    ``sidescan-*`` (sub)commands too — unrecognized keys are silently
    ignored here, not an error, so one project-wide per-dataset config
    file can be reused across every command without per-command
    filtering by the caller.

    Keys matching no option are logged as a WARNING naming the subcommand
    (a typo'd key would otherwise silently do nothing). With
    ``strict=True`` (``--strict-config``) they are a ``parser.error``
    instead; since a shared config legitimately holds other commands'
    keys, strict mode is only meaningful for a single-purpose config.
    """
    dests = {action.dest for action in parser._actions}
    unknown = sorted(k for k in config if k not in dests and k not in known_extra)
    if unknown and check:
        name = subcommand or parser.prog
        if strict:
            parser.error(f"--strict-config: unknown config key(s) for {name!r}: {', '.join(unknown)}")
        warn_deferred(logger, "config keys not recognized by %r (ignored): %s", name, ", ".join(unknown))
    parser.set_defaults(**{k: v for k, v in config.items() if k in dests})


def rename_config_keys(config: dict, renamed: "dict[str, str]") -> dict:
    """Return ``config`` with deprecated keys mapped to their current names,
    logging a WARNING for each. A key given under both names keeps the
    current one."""
    out = dict(config)
    for old, new in renamed.items():
        if old in out:
            warn_deferred(logger, "config key %r is deprecated; use %r", old, new)
            value = out.pop(old)
            out.setdefault(new, value)
    return out


def apply_config_to_subparsers(
    parsers: "list[argparse.ArgumentParser] | tuple[argparse.ArgumentParser, ...]",
    config: dict,
    argv: list[str],
    *,
    renamed_keys: "dict[str, str] | None" = None,
) -> None:
    """Seed every subparser from a shared ``config``, then report keys
    that match *no* subparser's options (a typo, as opposed to a key
    meant for a sibling subcommand, which a shared file legitimately
    holds). The message names the subcommand invoked in ``argv``.
    ``renamed_keys`` maps deprecated key names to current ones.

    Warns via ``logging``; ``--strict-config`` in ``argv`` makes it an error.
    """
    if renamed_keys:
        config = rename_config_keys(config, renamed_keys)
    for p in parsers:
        apply_config_defaults(p, config, check=False)
    names = {p.prog.split()[-1]: p for p in parsers}
    chosen = next((names[a] for a in argv if a in names), None)
    if chosen is not None:
        all_dests = {a.dest for p in parsers for a in p._actions}
        apply_config_defaults(
            chosen, config, strict=has_strict_flag(argv), subcommand=chosen.prog.split()[-1], known_extra=all_dests
        )
