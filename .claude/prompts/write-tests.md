
# TDD Workflow

## Principles

1. **Test first, code second**
2. **One assertion per test** (usually)
3. **Descriptive test names:** `test_<what>_<when>()`
4. **Fixtures for setup:** Use pytest fixtures, avoid duplication
5. **Group by feature:** Use test classes

## Workflow

1. **Understand the requirement** — What should this code do?
2. **Write failing test first** — Red
3. **Show test to user** — Get approval
4. **Implement minimum code to pass** — Green
5. **Run test** — Verify pass
6. **Refactor if needed** — Keep tests green
7. **Suggest commit** — After each logical addition

## Test File Structure

```python
"""
Unit tests for <module>.
"""

import pytest
import numpy as np
from sidescanpy.<module> import <thing>


@pytest.fixture
def sample_data():
    """Reusable test data."""
    return ...


class Test<Feature>:
    """Group of related tests."""

    def test_<what>_<when>(self, sample_data):
        """One-line description of what this tests."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...
```

## Common Assertions

```
# Shape/type
assert result.shape == (10, 20)
assert isinstance(result, np.ndarray)

# Values
assert result == expected
np.testing.assert_array_equal(a, b)
np.testing.assert_allclose(a, b, rtol=1e-5)

# Exceptions
with pytest.raises(ValueError, match="expected message"):
    function_that_should_fail()

# Skip conditions
pytest.skip("Reason for skipping")
```

## Running Tests

```
# All unit tests
pytest tests/unit/ -v

# Specific file
pytest tests/unit/grids/test_resample.py -v

# Specific test
pytest tests/unit/grids/test_resample.py::TestClass::test_method -v

# With coverage
pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing

# Stop on first failure
pytest tests/unit/ -x

# Show local variables on failure
pytest tests/unit/ -l
```