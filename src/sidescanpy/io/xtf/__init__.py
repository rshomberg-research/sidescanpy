"""XTF (eXtended Triton Format) reader -- stub package.

An XTF reader is planned but not yet implemented; see
``docs/_internal/xtf-reader.md`` for the design, references and test-data
sources. Only the public entry point exists so that dispatch code has a
stable name to target.
"""

from sidescanpy.io.xtf.reader import read_xtf

__all__ = ["read_xtf"]
