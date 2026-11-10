.PHONY: test test-v test-cov lint docs docs-fast docs-serve docs-clean readme-images

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
# without the dev extra, and the gallery needs viz and geo.
# `make docs` runs the gallery scripts on the public recordings (downloads
# about 1.3 GB the first time, 1.9 GB on disk); `make docs-fast` skips running them.
SPHINX = uv run --all-extras sphinx-build -W --keep-going -n -b html docs docs/_build/html

docs:
	$(SPHINX)

docs-fast:
	$(SPHINX) -D plot_gallery=0

docs-serve: docs
	uv run --all-extras python -m http.server -d docs/_build/html 8000

docs-clean:
	rm -rf docs/_build docs/examples docs/theory docs/reference/generated docs/sg_execution_times.rst

# Refresh the two gallery figures the README shows (after `make docs`).
readme-images:
	uv run --all-extras python -c "from PIL import Image; [Image.open(f'docs/examples/images/sphx_glr_plot_{n}_001.png').convert('RGB').quantize(256, dither=Image.Dither.NONE).save(f'docs/assets/gallery/{a}.png', optimize=True) for n, a in (('correct_a_line', 'correct_a_line'), ('compare_sonars_and_formats', 'three_formats'))]"
