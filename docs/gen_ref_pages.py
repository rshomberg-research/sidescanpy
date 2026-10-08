"""Generate one API reference page per public module (run by mkdocs-gen-files).

Pages go to ``reference/`` in the built site, with ``reference/SUMMARY.md``
as the navigation (mkdocs-literate-nav). Modules and packages whose name
starts with ``_`` are private and skipped, as is ``sidescanpy.cli`` (its
commands are documented in ``cli.md``). Package pages show the package
docstring only; each function and class is documented once, on the page
of the module that defines it.
"""

from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).resolve().parent.parent / "src"
nav = mkdocs_gen_files.Nav()

for path in sorted((SRC / "sidescanpy").rglob("*.py")):
    module = path.relative_to(SRC).with_suffix("")
    parts = list(module.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        is_package = True
    else:
        is_package = False
    if any(p.startswith("_") for p in parts) or (len(parts) > 1 and parts[1] == "cli"):
        continue
    doc_path = Path(*parts[1:]).with_suffix(".md") if len(parts) > 1 else Path("index.md")
    if is_package and len(parts) > 1:
        doc_path = Path(*parts[1:], "index.md")
    nav[tuple(parts[1:]) or ("sidescanpy",)] = doc_path.as_posix()
    identifier = ".".join(parts)
    with mkdocs_gen_files.open(Path("reference", doc_path), "w") as f:
        f.write(f"# `{identifier}`\n\n::: {identifier}\n")
        if is_package:
            f.write("    options:\n      members: false\n")
    mkdocs_gen_files.set_edit_path(Path("reference", doc_path), path.relative_to(SRC.parent))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
