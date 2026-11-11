"""Small Sphinx helpers for the sidescanpy docs.

- ``sidescanpy_noindex``: when true, every page gets
  ``<meta name="robots" content="noindex, nofollow">``, to keep a site out of
  search engines.
- Python cross-references that don't resolve as written are retried by
  name: ``:func:`read_and_convert``` or ``sidescanpy.io.read_and_convert``
  (a re-export) link to the one documented object of that name. Names
  that match several objects stay unlinked.
- ``gallery_argv``: the sphinx-gallery ``reset_argv`` callable.
- ``gallery_order``: the galleries' teaching order (``within_subsection_order``).
- Goal image first: on every gallery page (Examples and Theory) the
  page's result figure is shown under the title, before the text and the
  code that makes it. By default that is the page's last figure;
  :data:`GOAL_IMAGE` picks another.
- Gallery thumbnails: each example writes a clean thumbnail image into
  its output folder and names it with ``sphinx_gallery_thumbnail_path``.
  When the gallery is not executed (``-D plot_gallery=0``) those files
  don't exist; only that warning is silenced then, so a full build stays
  strict.
"""

import os
import re
from pathlib import Path

from sphinx.addnodes import pending_xref


def gallery_argv(gallery_conf, script_vars):
    """sphinx-gallery ``reset_argv``: the scripts' output directory.

    The gallery scripts write their ``.ss.h5``, GeoTIFF and PNG output to
    the directory given as their first argument; this puts it under
    ``docs/_build/`` instead of the source tree.
    """
    out = Path(gallery_conf["src_dir"]) / "_build" / "gallery-output"
    out.mkdir(parents=True, exist_ok=True)
    return [str(out)]


GALLERY_ORDER = [
    "plot_read_a_file",
    "plot_whats_in_a_file",
    "plot_navigation",
    "plot_investigate_a_survey",
    "plot_correct_a_line",
    "plot_compare_sonars_and_formats",
    "plot_grid_to_a_map",
    "plot_wreck_gallery",
    "plot_four_passes",
    "plot_swath_tiles",
    "plot_target_chips",
    "plot_training_dataset",
    "plot_vendor_chain",
    "plot_four_published_chains",
    # Theory, in reading order
    "plot_how_sidescan_sees",
    "plot_amplitude_and_db",
    "plot_range_limits",
    "plot_frequency_and_resolution",
    "plot_sonar_equation_and_tvg",
    "plot_slant_to_ground_range",
    "plot_slopes_and_layover",
    "plot_shadow_height",
    "plot_ping_spacing_and_coverage",
    "plot_towfish_attitude",
    "plot_navigation_and_position",
    "plot_common_artifacts",
]


#: Gallery pages whose goal image is not their last figure: script stem ->
#: figure number (1-based, as in ``sphx_glr_<stem>_001.png``).
GOAL_IMAGE: dict[str, int] = {"plot_common_artifacts": 1, "plot_correct_a_line": 1, "plot_grid_to_a_map": 1}

_GALLERY_DOCS = ("examples/plot_", "theory/plot_")
_IMAGE_BLOCK = re.compile(r"^\.\. image-sg:: \S+\n(?:[ \t]+\S.*\n)*", re.M)
_TITLE = re.compile(r"^\.\. _sphx_glr_\S+:\n\s*\n[^\n]+\n=+\n", re.M)


def _goal_image_first(app, docname, source):
    """Show a gallery page's result figure under its title (``source-read``)."""
    if not docname.startswith(_GALLERY_DOCS):
        return
    text = source[0]
    blocks = _IMAGE_BLOCK.findall(text)
    title = _TITLE.search(text)
    if not blocks or title is None:
        return  # gallery not executed, or not a sphinx-gallery page
    wanted = GOAL_IMAGE.get(Path(docname).name)
    block = next((b for b in blocks if wanted and f"_{wanted:03d}.png" in b.split("\n", 1)[0]), blocks[-1])
    source[0] = text[: title.end()] + "\n" + block + "\n" + text[title.end():]


def gallery_order(filename):
    """sphinx-gallery ``within_subsection_order``: examples in teaching order.

    Files not in :data:`GALLERY_ORDER` follow, alphabetically.
    """
    stem = Path(filename).stem
    return (GALLERY_ORDER.index(stem), stem) if stem in GALLERY_ORDER else (len(GALLERY_ORDER), stem)


_NOINDEX = '<meta name="robots" content="noindex, nofollow" />\n'


def _add_noindex(app, pagename, templatename, context, doctree):
    if app.config.sidescanpy_noindex:
        context["metatags"] = context.get("metatags", "") + _NOINDEX


def _resolve_by_name(app, env, node: pending_xref, contnode):
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "").lstrip(".~").removesuffix("()")
    if not target:
        return None
    domain = env.get_domain("py")
    objtypes = set(domain.objtypes_for_role(node.get("reftype")) or [])
    name = target.rsplit(".", 1)[-1]
    candidates = [
        (fullname, obj)
        for fullname, obj in domain.objects.items()
        if fullname.startswith("sidescanpy.")
        and not obj.aliased
        and (not objtypes or obj.objtype in objtypes)
        and (fullname.endswith("." + target) or fullname.rsplit(".", 1)[-1] == name)
    ]
    if target.startswith("sidescanpy."):
        # A re-export: sidescanpy.io.read_and_convert is documented as
        # sidescanpy.io.dispatch.read_and_convert.
        package = target.rsplit(".", 1)[0] + "."
        candidates = [c for c in candidates if c[0].startswith(package)]
    elif "." in target:
        # Class.method
        candidates = [c for c in candidates if c[0].endswith("." + target)]
    if len(candidates) > 1:
        # Prefer objects in or below the module the reference is written in.
        here = node.get("py:module") or ""
        while here and len(candidates) > 1:
            narrowed = [c for c in candidates if c[0].startswith(here + ".")]
            if narrowed:
                candidates = narrowed
                break
            here = here.rpartition(".")[0]
    if len(candidates) != 1:
        return None
    fullname, obj = candidates[0]
    return domain.resolve_xref(env, node["refdoc"], app.builder, obj.objtype,
                               fullname, node, contnode)


def _quiet_thumbnails_when_not_executed(app, config):
    if str(getattr(config, "plot_gallery", "True")).strip().lower() in ("0", "false", "no", "off"):
        config.suppress_warnings = [*config.suppress_warnings, "sphinx_gallery.thumbnail"]


#: Pages that moved: old path (no suffix) -> new path. The old URL keeps
#: working as a small page that redirects to the new one.
MOVED_PAGES = {
    "formats": "sources/index",
    "io_jsf": "sources/jsf",
    "jsf_aux_source_selection": "sources/jsf",
    "io_sdf": "sources/sdf",
    "io_xtf": "sources/xtf",
    "io_hdf5": "format/ss_h5",
    "io_netcdf": "format/netcdf",
    "examples/plot_shadow_height": "theory/plot_shadow_height",
    "examples/plot_common_artifacts": "theory/plot_common_artifacts",
}

_REDIRECT = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Moved</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
</head><body><p>This page moved to <a href="{target}">{target}</a>.</p></body></html>
"""


def _write_redirects(app, exception):
    """Write a redirect page at each old URL in :data:`MOVED_PAGES` (HTML builds only)."""
    if exception is not None or app.builder.format != "html":
        return
    out = Path(app.outdir)
    for old, new in MOVED_PAGES.items():
        target = os.path.relpath(new, Path(old).parent) + ".html"
        (out / f"{old}.html").write_text(_REDIRECT.format(target=target.replace(os.sep, "/")))


def setup(app):
    app.connect("source-read", _goal_image_first)
    app.connect("build-finished", _write_redirects)
    app.add_config_value("sidescanpy_noindex", False, "html")
    app.connect("html-page-context", _add_noindex)
    app.connect("missing-reference", _resolve_by_name)
    app.connect("config-inited", _quiet_thumbnails_when_not_executed)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
