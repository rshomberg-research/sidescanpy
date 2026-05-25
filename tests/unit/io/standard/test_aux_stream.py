"""
Unit tests for auxiliary stream I/O

2026 rshomberg@uri.edu
"""

import pytest
import numpy as np
import h5py
from pathlib import Path

from sidescanpy.io.standard.aux_streams import (
    write_navigation_stream,
    read_navigation_stream,
    write_attitude_stream,
    read_attitude_stream,
    write_environmental_stream,
    read_environmental_stream,
    interpolate_to_ping_time,
)


@pytest.mark.unit
class TestNavigationStream:
    """Test navigation stream read/write"""

    def test_write_navigation_creates_group(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        n_fixes = 100
        time = np.linspace(0, 1000, n_fixes)
        lat = np.linspace(40.0, 40.5, n_fixes)
        lon = np.linspace(-70.0, -69.9, n_fixes)
        head = np.random.rand(n_fixes) * 360
        speed = np.full(n_fixes, 5.0)
        depth = np.random.rand(n_fixes) * 200

        write_navigation_stream(output_path, time, lat, lon, head, speed, depth)

        with h5py.File(output_path, "r") as f:
            assert "aux_streams/navigation" in f  # type: ignore

    def test_write_read_navigation_preserves_data(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        n_fixes = 100
        time = np.linspace(0, 1000, n_fixes)
        lat = np.linspace(40.0, 40.5, n_fixes)
        lon = np.linspace(-70.0, -69.9, n_fixes)
        head = np.random.rand(n_fixes) * 360
        speed = np.full(n_fixes, 5.0)
        depth = np.random.rand(n_fixes) * 200

        write_navigation_stream(output_path, time, lat, lon, head, speed, depth)
        time_read, data_read = read_navigation_stream(output_path)

        np.testing.assert_array_almost_equal(time_read, time)
        np.testing.assert_array_almost_equal(data_read["latitude"], lat)
        np.testing.assert_array_almost_equal(data_read["longitude"], lon)


@pytest.mark.unit
class TestAttitudeStream:
    """Test attitude stream read/write"""

    def test_write_attitude_creates_group(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        n_samples = 500
        time = np.linspace(0, 1000, n_samples)
        roll = np.random.randn(n_samples) * 5
        pitch = np.random.randn(n_samples) * 5
        heave = np.random.randn(n_samples) * 2
        altitude = np.random.rand(n_samples) * 100 + 50

        write_attitude_stream(output_path, time, roll, pitch, heave, altitude)

        with h5py.File(output_path, "r") as f:
            assert "aux_streams/attitude" in f  # type: ignore

    def test_write_read_attitude_preserves_data(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        n_samples = 500
        time = np.linspace(0, 1000, n_samples)
        roll = np.random.randn(n_samples) * 5
        pitch = np.random.randn(n_samples) * 5
        heave = np.random.randn(n_samples) * 2
        altitude = np.random.rand(n_samples) * 100 + 50

        write_attitude_stream(output_path, time, roll, pitch, heave, altitude)
        time_read, data_read = read_attitude_stream(output_path)

        np.testing.assert_array_almost_equal(time_read, time)
        np.testing.assert_array_almost_equal(data_read["roll_deg"], roll)
        np.testing.assert_array_almost_equal(data_read["altitude_m"], altitude)


@pytest.mark.unit
class TestEnvironmentalStream:
    """Test environmental stream read/write"""

    def test_write_environmental_creates_group(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        time = np.array([0.0])  # Single measurement
        sound_vel = np.array([1500.0])

        write_environmental_stream(output_path, time, sound_vel)

        with h5py.File(output_path, "r") as f:
            assert "aux_streams/environmental" in f  # type: ignore

    def test_write_read_environmental_preserves_data(self, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        time = np.array([0.0, 500.0, 1000.0])
        sound_vel = np.array([1500.0, 1498.0, 1502.0])

        write_environmental_stream(output_path, time, sound_vel)
        time_read, data_read = read_environmental_stream(output_path)

        np.testing.assert_array_almost_equal(time_read, time)
        np.testing.assert_array_almost_equal(data_read["sound_vel_ms"], sound_vel)


@pytest.mark.unit
class TestInterpolation:
    """Test interpolation to ping times"""

    def test_interpolate_linear(self):
        source_time = np.array([0.0, 100.0])
        source_data = np.array([0.0, 100.0])
        ping_time = np.array([0.0, 50.0, 100.0])

        result = interpolate_to_ping_time(source_time, source_data, ping_time, method="linear")

        np.testing.assert_array_almost_equal(result, [0.0, 50.0, 100.0])

    def test_interpolate_extrapolate_nearest(self):
        """Out-of-bounds should use nearest value"""
        source_time = np.array([10.0, 20.0])
        source_data = np.array([10.0, 20.0])
        ping_time = np.array([5.0, 15.0, 25.0])

        result = interpolate_to_ping_time(source_time, source_data, ping_time, method="nearest")

        # Should have valid values at all ping times
        assert result.shape == ping_time.shape
        assert not np.any(np.isnan(result))

    # TODO: test interpolate with extrapolate
    # TODO: test interpolate with other methods
    # ToDO: test multi-dimensional interpolation