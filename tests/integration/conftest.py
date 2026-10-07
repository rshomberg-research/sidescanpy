"""Fixtures for integration tests"""

import pytest

from pathlib import Path 

@pytest.fixture
def small_jsf_file():
    return Path(__file__).resolve().parents[1] / "data" / "jsf" / "sample.jsf"