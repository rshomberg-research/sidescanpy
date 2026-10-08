"""
Logging helpers for sidescanpy.

Policy: library modules only create ``logger = logging.getLogger(__name__)``
and log at DEBUG (parameters used, defaults applied, fallbacks) or WARNING
(real data problems). They never configure handlers and never print.
Entry points (the ``sidescan-*`` CLIs, consumer scripts) call
:func:`configure_logging` once to decide what is shown and where.

2026 rshomberg@uri.edu
"""

import contextlib
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Marker attribute set on handlers installed by configure_logging so a
# second call can find and replace them instead of stacking duplicates.
_HANDLER_TAG = "_sidescanpy_configured"

_NOISY_LOGGERS = ("matplotlib", "PIL", "h5py", "fiona", "rasterio", "urllib3")


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).upper())
    if not isinstance(value, int):
        raise ValueError(f"Unknown logging level: {level!r}")
    return value


def configure_logging(
    level: int | str = "INFO",
    *,
    log_file: str | Path | None = None,
    library_level: int | str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure the root logger for a command-line or script entry point.

    Installs a stderr stream handler (and a file handler if ``log_file`` is
    given) on the root logger at ``level``. ``library_level`` sets the
    ``sidescanpy`` logger's own level (e.g. ``"DEBUG"`` for ``-v``) while
    leaving third-party loggers at ``level``; ``None`` lets it inherit.
    Noisy third-party loggers are always held at WARNING. Safe to call
    repeatedly: handlers from a previous call are replaced.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(fmt or LOG_FORMAT, datefmt=LOG_DATEFMT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    root_level = _coerce_level(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    # The root logger must pass records at the more verbose of the two
    # levels, so a DEBUG library_level is not blocked upstream; handlers
    # then filter third-party records back down to ``level``.
    lib_logger = logging.getLogger("sidescanpy")
    if library_level is None:
        lib_logger.setLevel(logging.NOTSET)
        root.setLevel(root_level)
    else:
        lib_level = _coerce_level(library_level)
        lib_logger.setLevel(lib_level)
        root.setLevel(root_level)
        # Records from sidescanpy reach root handlers regardless of the root
        # *logger* level (only handler levels apply on propagation), so give
        # handlers a floor of the lower of the two levels and filter others.
        floor = min(root_level, lib_level)
        for handler in handlers:
            handler.setLevel(floor)
            if floor < root_level:
                handler.addFilter(_ThirdPartyLevelFilter(root_level))

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class _ThirdPartyLevelFilter(logging.Filter):
    """Hold non-sidescanpy records to ``min_level`` when handlers are
    opened wider for a more verbose library logger."""

    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "sidescanpy" or record.name.startswith("sidescanpy."):
            return True
        return record.levelno >= self.min_level


@contextlib.contextmanager
def log_timing(
    logger: logging.Logger, label: str, level: int = logging.DEBUG
) -> Iterator[None]:
    """Log ``"<label>: <seconds>s"`` when the ``with`` block exits."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s: %.2fs", label, time.perf_counter() - start)
