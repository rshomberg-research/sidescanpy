"""
Re-run the gallery examples and refresh the images the docs show.

Runs every ``examples/gallery/*.py`` with a temporary output directory
and copies each PNG it writes to ``docs/assets/gallery/`` as a 256-colour
palette image (about half the size; the sonar colour maps have fewer
colours than that). Needs network access the first time, or
``SIDESCANPY_DATA_DIR`` pointing at the downloaded recordings.

Usage:
    make gallery
    python docs/build_gallery.py [script_stem ...]

2026 rshomberg@uri.edu
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "examples" / "gallery"
ASSETS = ROOT / "docs" / "assets" / "gallery"


def main(stems: list[str]) -> int:
    scripts = sorted(GALLERY.glob("*.py"))
    if stems:
        scripts = [s for s in scripts if s.stem in stems]
    ASSETS.mkdir(parents=True, exist_ok=True)
    for script in scripts:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"running {script.relative_to(ROOT)}", flush=True)
            subprocess.run([sys.executable, str(script), tmp], check=True)
            for png in sorted(Path(tmp).glob("*.png")):
                image = Image.open(png).convert("RGB").quantize(256, dither=Image.Dither.NONE)
                image.save(ASSETS / png.name, optimize=True)
                print(f"  wrote docs/assets/gallery/{png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
