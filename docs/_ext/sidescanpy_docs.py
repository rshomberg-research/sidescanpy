"""Small Sphinx helpers for the sidescanpy docs.

- ``sidescanpy_noindex``: when true, every page gets
  ``<meta name="robots" content="noindex, nofollow">``, to keep a site out of
  search engines.
- Python cross-references that don't resolve as written are retried by
  name: ``:func:`read_and_convert``` or ``sidescanpy.io.read_and_convert``
  (a re-export) link to the one documented object of that name. Names
  that match several objects stay unlinked.
- ``gallery_argv``: the sphinx-gallery ``reset_argv`` callable.
- ``gallery_order``: the gallery's teaching order (``within_subsection_order``).
- Gallery thumbnails: each example writes a clean thumbnail image into
  its output folder and names it with ``sphinx_gallery_thumbnail_path``.
  When the gallery is not executed (``-D plot_gallery=0``) those files
  don't exist; only that warning is silenced then, so a full build stays
  strict.
"""

import os
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
    "plot_common_artifacts",
    "plot_correct_a_line",
    "plot_compare_sonars_and_formats",
    "plot_grid_to_a_map",
    "plot_wreck_gallery",
    "plot_four_passes",
    "plot_shadow_height",
    "plot_swath_tiles",
    "plot_target_chips",
    "plot_training_dataset",
]


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
    app.connect("build-finished", _write_redirects)
    app.add_config_value("sidescanpy_noindex", False, "html")
    app.connect("html-page-context", _add_noindex)
    app.connect("missing-reference", _resolve_by_name)
    app.connect("config-inited", _quiet_thumbnails_when_not_executed)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
