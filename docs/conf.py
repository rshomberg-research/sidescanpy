"""Sphinx configuration for the sidescanpy documentation.

Guide pages are MyST Markdown in ``docs/``. The API reference is
generated from the numpydoc docstrings into
``docs/reference/generated/`` by ``_ext/api_pages.py``. The
examples and theory galleries are built by sphinx-gallery from
``examples/gallery/`` and ``examples/theory/`` into ``docs/examples/`` and
``docs/theory/``, running every script on the public recordings
(``plot_gallery=0`` skips running them).
"""

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS / "_ext"))

import sidescanpy  # noqa: E402

project = "sidescanpy"
author = "Russ Shomberg"
copyright = "2026, Russ Shomberg"
release = sidescanpy.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sidescanpy_docs",
    "api_pages",
]

exclude_patterns = [
    "_build",
    "examples/*.ipynb",
    "examples/*.py",
    "examples/*.md5",
    "examples/*.zip",
    "examples/*.codeobj.json",
    "assets/datasets/*_caption.md",  # included by public_datasets.md
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
# Generate anchors for headings so links like ``processing.md#tvg`` resolve.
myst_heading_anchors = 3

# -- API reference -----------------------------------------------------------
# _ext/api_pages.py writes the module pages; autosummary only renders the
# summary tables on them.
autosummary_generate = False
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "none"  # numpydoc Parameters sections carry the types
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False
# Single backticks in the docstrings mean inline code, as in numpydoc.
default_role = "literal"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable", None),
}

# -- Gallery -----------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/gallery", "../examples/theory"],
    "gallery_dirs": ["examples", "theory"],
    "filename_pattern": r"/plot_",
    # Each script gets a scratch output directory as its first argument.
    "reset_argv": "sidescanpy_docs.gallery_argv",
    "within_subsection_order": "sidescanpy_docs.gallery_order",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_signature": False,
    "matplotlib_animations": False,
    "abort_on_example_error": True,
    # 2x images for high-resolution screens (the chip and tile figures are
    # detailed).
    "image_srcset": ["2x"],
}

# -- HTML --------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"sidescanpy {release}"
html_static_path = []
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/rshomberg-research/sidescanpy",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "header_links_before_dropdown": 7,
    "secondary_sidebar_items": ["page-toc"],
}
html_sidebars = {"index": [], "install": [], "getting_started": [], "status": []}

# True adds a noindex meta tag to every page (sidescanpy_docs extension),
# keeping a site out of search engines.
sidescanpy_noindex = False
