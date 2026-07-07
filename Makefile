.PHONY: test test-v test-cov

# ROS2 injects PYTHONPATH system-wide; clear it so uv run sees only the venv.
# Long-term fix: conda env for ROS, uv for this project.
PYTHONPATH =

test:
	uv run pytest tests/unit/ -q

test-v:
	uv run pytest tests/unit/ -v

test-cov:
	uv run pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing
