.PHONY: test test-v test-cov lint docs docs-serve gallery

# ROS2 injects PYTHONPATH system-wide; clear it so uv run sees only the venv.
# Long-term fix: conda env for ROS, uv for this project.
PYTHONPATH =

test:
	uv run pytest tests/unit/ -q

test-v:
	uv run pytest tests/unit/ -v

test-cov:
	uv run pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing

lint:
	uv run ruff check src tests

# --all-extras: `uv run --extra docs` alone would re-sync the environment
# without the dev extra. ruff (dev) also formats the rendered signatures.
DOCS_ENV = DISABLE_MKDOCS_2_WARNING=true NO_MKDOCS_2_WARNING=1

docs:
	$(DOCS_ENV) uv run --all-extras mkdocs build --strict

docs-serve:
	$(DOCS_ENV) uv run --all-extras mkdocs serve

# Re-run examples/gallery/ on the public recordings and refresh the
# images in docs/assets/gallery/ (downloads ~140 MB the first time).
gallery:
	uv run --all-extras python docs/build_gallery.py
