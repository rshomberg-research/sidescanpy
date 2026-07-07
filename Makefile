.PHONY: test test-v test-cov

# ROS2 injects PYTHONPATH system-wide; clear it so pytest sees only the venv.
# Use 'conda activate sidescanpy' as the long-term fix.
PYTHONPATH =

test:
	python -m pytest tests/unit/ -q

test-v:
	python -m pytest tests/unit/ -v

test-cov:
	python -m pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing
