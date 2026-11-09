"""Write one API reference page per public sidescanpy module.

Run at ``builder-inited``. Pages go to ``docs/reference/generated/``:

- a package page shows the package docstring, its own functions and
  classes (if any) and a table of contents of its subpackages and
  modules;
- a module page shows the module docstring, an autosummary table of what
  it defines, then each constant, class and function in source order.

Each object is documented once, on the page of the module that defines
it (re-exports are skipped), except that a name a package lists in
``__all__`` from one of its private modules is documented on the
package page. Names starting with ``_`` and the
``sidescanpy.cli`` package (documented in ``cli.md``) are left out.

This walks the modules directly instead of using autosummary's
``:recursive:`` mode, because some packages re-export a function with the
same name as one of their modules (``sidescanpy.processing.correct``),
which autosummary resolves to the function.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

PACKAGE = "sidescanpy"
SKIP = {"sidescanpy.cli"}
# A constant whose source is longer than this is shown without its value.
MAX_VALUE_CHARS = 60


def _public_modules():
    root = importlib.import_module(PACKAGE)
    names = [PACKAGE]
    for info in pkgutil.walk_packages(root.__path__, PACKAGE + "."):
        parts = info.name.split(".")
        if any(p.startswith("_") for p in parts):
            continue
        if any(info.name == s or info.name.startswith(s + ".") for s in SKIP):
            continue
        names.append(info.name)
    return sorted(names)


def _definitions(module):
    """(kind, name, show_value) of the public objects a module defines, in source order."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return []
    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            if name.startswith("_") or not hasattr(module, name):
                continue
            obj = getattr(module, name)
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if isinstance(node, ast.ClassDef):
                kind = "exception" if isinstance(obj, type) and issubclass(obj, BaseException) else "class"
            else:
                kind = "function"
            out.append((kind, name, True))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_"):
                    seg = ast.get_source_segment(source, value) if value is not None else ""
                    out.append(("data", t.id, seg is not None and len(seg) <= MAX_VALUE_CHARS))
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        origin = getattr(obj, "__module__", "") or ""
        if (origin.startswith(module.__name__ + "._")
                and (inspect.isfunction(obj) or inspect.isclass(obj))):
            out.append(("class" if inspect.isclass(obj) else "function", name, True))
    return out


def _page(modname, children):
    module = importlib.import_module(modname)
    title = modname
    lines = [title, "=" * len(title), "", f".. automodule:: {modname}", ""]
    defs = _definitions(module)
    if defs:
        lines += [f".. currentmodule:: {modname}", "", ".. rubric:: Contents", "",
                  ".. autosummary::", "   :nosignatures:", ""]
        lines += [f"   {name}" for _, name, _ in defs]
        lines.append("")
    if children:
        lines += [".. rubric:: Modules", "", ".. toctree::", "   :maxdepth: 1", ""]
        lines += [f"   {c.rsplit('.', 1)[-1]} <{c}>" for c in children]
        lines.append("")
    for kind, name, show_value in defs:
        full = f"{modname}.{name}"
        if kind == "data":
            lines.append(f".. autodata:: {full}")
            if not show_value:
                lines.append("   :no-value:")
        elif kind == "class":
            lines += [f".. autoclass:: {full}", "   :members:", "   :show-inheritance:"]
        elif kind == "exception":
            lines += [f".. autoexception:: {full}", "   :show-inheritance:"]
        else:
            lines.append(f".. autofunction:: {full}")
        lines.append("")
    return "\n".join(lines)


def write_pages(app):
    out = Path(app.srcdir) / "reference" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    modules = _public_modules()
    wanted = set()
    for modname in modules:
        children = [m for m in modules if m != modname and m.rsplit(".", 1)[0] == modname]
        text = _page(modname, children)
        path = out / f"{modname}.rst"
        wanted.add(path.name)
        if not path.exists() or path.read_text() != text:  # keep mtimes for incremental builds
            path.write_text(text)
    for stale in out.glob("*.rst"):
        if stale.name not in wanted:
            stale.unlink()


def _drop_private_bases(app, name, obj, options, bases):
    bases[:] = [b for b in bases if not getattr(b, "__name__", "").startswith("_")
                and not any(p.startswith("_") for p in getattr(b, "__module__", "").split("."))]


def setup(app):
    app.connect("builder-inited", write_pages)
    app.connect("autodoc-process-bases", _drop_private_bases)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
