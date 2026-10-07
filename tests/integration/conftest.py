"""Fixtures for integration tests"""

from pathlib import Path

import pytest


@pytest.fixture
def small_jsf_file():
    return Path(__file__).resolve().parents[1] / "data" / "jsf" / "sample.jsf"
