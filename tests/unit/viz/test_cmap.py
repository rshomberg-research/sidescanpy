"""
2026 rshomberg@uri.edu
"""

import numpy as np
import pytest

from sidescanpy.viz._cmap import resolve_cmap


class TestResolveCmap:
    def test_nan_renders_as_nan_color(self):
        cmap = resolve_cmap("viridis")
        rgba = cmap(np.nan)
        assert rgba == pytest.approx(cmap.get_bad())

    def test_bad_color_is_black(self):
        cmap = resolve_cmap("copper")
        assert cmap.get_bad() == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_finite_values_unaffected(self):
        name = "viridis"
        import matplotlib.pyplot as plt

        original = plt.get_cmap(name)
        resolved = resolve_cmap(name)
        assert resolved(0.5) == original(0.5)

    def test_accepts_colormap_instance_not_just_name(self):
        import matplotlib.pyplot as plt

        instance = plt.get_cmap("copper")
        resolved = resolve_cmap(instance)
        assert resolved.get_bad() == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_does_not_mutate_shared_registry_colormap(self):
        import matplotlib.pyplot as plt

        before = plt.get_cmap("viridis").get_bad()
        resolve_cmap("viridis")
        after = plt.get_cmap("viridis").get_bad()
        np.testing.assert_array_equal(before, after)
