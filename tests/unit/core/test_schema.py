"""
Test module for SideScanPy core schema validation and xr.Accessor

2026 rshomberg@uri.edu
"""

import pytest 
import json 
import numpy as np 
import xarray as xr 
from datetime import datetime, timezone

from sidescanpy.core.schema import (
    REQUIRED_DIMS,
    REQUIRED_COORDS,
    REQUIRED_VARS,
    REQUIRED_ATTRS, 
    KNOWN_PROCESSING_LEVELS,
    HDF5_GROUP_AUX_STREAMS,
    HDF5_GROUP_NAVIGATION,
    HDF5_GROUP_ATTITUDE,
    HDF5_GROUP_FREQUENCIES,
    HDF5_FREQ_PREFIX,
    HDF5_CHUNK_PING,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_LEVEL,    
    REQUIRED_NAV_DIMS,
    REQUIRED_NAV_COORDS,
    REQUIRED_NAV_VARS,
    REQUIRED_NAV_ATTRS,
    REQUIRED_ATT_DIMS,
    REQUIRED_ATT_COORDS,
    REQUIRED_ATT_VARS,
    REQUIRED_ATT_ATTRS,
    REQUIRED_DVL_DIMS,
    REQUIRED_DVL_COORDS,
    REQUIRED_DVL_VARS,
    REQUIRED_DVL_ATTRS,
    REQUIRED_GEN_DIMS,
    REQUIRED_GEN_COORDS,
    REQUIRED_GEN_ATTRS,
)

@pytest.mark.unit 
class TestConstants:
    """Verify schema constants are correctly defined"""

    def test_required_dims_correct(self):
        assert isinstance(REQUIRED_DIMS, set)
        assert REQUIRED_DIMS == {"ping", "channel", "sample"}

    def test_required_coords_correct(self):
        assert isinstance(REQUIRED_COORDS, set)
        assert REQUIRED_COORDS == {"ping_number", "time", "channel_id"}

    def test_required_vars_correct(self):
        assert isinstance(REQUIRED_VARS, set)
        assert "backscatter_raw" in REQUIRED_VARS
        assert "backscatter_intensity" in REQUIRED_VARS
        assert "latitude" in REQUIRED_VARS
        assert "longitude" in REQUIRED_VARS
        assert "heading_deg" in REQUIRED_VARS
        assert "depth_m" in REQUIRED_VARS
        assert "roll_deg" in REQUIRED_VARS
        assert "pitch_deg" in REQUIRED_VARS
        assert "heave_m" in REQUIRED_VARS
        assert "altitude_m" in REQUIRED_VARS
        assert "sound_vel_ms" in REQUIRED_VARS

    def test_required_attrs_correct(self):
        assert isinstance(REQUIRED_ATTRS, set)
        assert "freq_hz_start" in REQUIRED_ATTRS
        assert "freq_hz_end" in REQUIRED_ATTRS
        assert "pulse_length_us" in REQUIRED_ATTRS
        assert "sample_rate_hz" in REQUIRED_ATTRS
        assert "processing_level" in REQUIRED_ATTRS
        assert "processing_history" in REQUIRED_ATTRS

    def test_known_processing_levels_correct(self):
        assert isinstance(KNOWN_PROCESSING_LEVELS, set)
        assert KNOWN_PROCESSING_LEVELS == {"raw", "radiometric"}

    def test_hdf5_constants_correct(self):
        assert isinstance(HDF5_GROUP_AUX_STREAMS, str)
        assert isinstance(HDF5_GROUP_NAVIGATION, str)
        assert isinstance(HDF5_GROUP_ATTITUDE, str)
        # assert isinstance(HDF5_GROUP_ENVIRONMENTAL, str)
        assert isinstance(HDF5_GROUP_FREQUENCIES, str)
        assert isinstance(HDF5_FREQ_PREFIX, str)
        assert isinstance(HDF5_CHUNK_PING, int)
        assert isinstance(HDF5_COMPRESSION, str)
        assert isinstance(HDF5_COMPRESSION_LEVEL, int)


@pytest.mark.unit 
class TestAccessorRegistration:
    """Verify xr.Accessor is registered and accessible"""

    def test_accessor_exists(self, sample_dataset):
        assert hasattr(sample_dataset, "sidescan")

    def test_validate_method_exists(self, sample_dataset):
        assert hasattr(sample_dataset.sidescan, "validate")
        assert callable(sample_dataset.sidescan.validate)

    def test_is_valid_method_exists(self, sample_dataset):
        assert hasattr(sample_dataset.sidescan, "is_valid")
        assert callable(sample_dataset.sidescan.is_valid)

    def test_processing_level_property_exists(self, sample_dataset):
        assert hasattr(sample_dataset.sidescan, "processing_level")

    def test_processing_history_property_exists(self, sample_dataset):
        assert hasattr(sample_dataset.sidescan, "processing_history")

    def test_append_to_history_method_exists(self, sample_dataset):
        assert hasattr(sample_dataset.sidescan, "append_to_history")
        assert callable(sample_dataset.sidescan.append_to_history)    


@pytest.mark.unit 
class TestValidationValid:
    """Test that valid datasets pass validation."""

    def test_valid_dataset_passes(self, sample_dataset):
        sample_dataset.sidescan.validate()  # Should not raise

    def test_valid_dataset_is_valid(self, sample_dataset):
        assert sample_dataset.sidescan.is_valid() is True


@pytest.mark.unit
class TestValidationMissingDims:
    """Test that missing dimensions raise ValueError."""

    @pytest.mark.parametrize("dim_name", REQUIRED_DIMS)
    def test_missing_dimension_raises(self, sample_dataset, dim_name):
        bad_ds = sample_dataset.drop_dims(dim_name)
        with pytest.raises(ValueError, match="Missing dimensions"):
            bad_ds.sidescan.validate()

    @pytest.mark.parametrize("dim_name", REQUIRED_DIMS)
    def test_missing_dimension_invalid(self, sample_dataset, dim_name):
        bad_ds = sample_dataset.drop_dims(dim_name)
        assert bad_ds.sidescan.is_valid() is False


@pytest.mark.unit
class TestValidationChannelSize:
    """Test that channel dimension must be exactly size 2."""

    def test_channel_size_not_2_raises(self, sample_dataset):
        bad_ds = sample_dataset.isel(channel=slice(0, 1))
        with pytest.raises(ValueError, match="'channel' dimension must be size 2"):
            bad_ds.sidescan.validate()

    def test_channel_size_not_2_invalid(self, sample_dataset):
        bad_ds = sample_dataset.isel(channel=slice(0, 1))
        assert bad_ds.sidescan.is_valid() is False


@pytest.mark.unit
class TestValidationMissingCoords:
    """Test that missing coordinates raise ValueError."""

    @pytest.mark.parametrize("coord_name", REQUIRED_COORDS)
    def test_missing_coordinate_raises(self, sample_dataset, coord_name):
        bad_ds = sample_dataset.drop_vars(coord_name)
        with pytest.raises(ValueError, match="Missing coordinates"):
            bad_ds.sidescan.validate()

    @pytest.mark.parametrize("coord_name", REQUIRED_COORDS)
    def test_missing_coordinate_invalid(self, sample_dataset, coord_name):
        bad_ds = sample_dataset.drop_vars(coord_name)
        assert bad_ds.sidescan.is_valid() is False

        
@pytest.mark.unit
class TestValidationMissingVars:
    """Test that missing variables raise ValueError."""

    @pytest.mark.parametrize("var_name", REQUIRED_VARS)
    def test_missing_variable_raises(self, sample_dataset, var_name):
        bad_ds = sample_dataset.drop_vars(var_name)
        with pytest.raises(ValueError, match="Missing variables"):
            bad_ds.sidescan.validate()

    @pytest.mark.parametrize("var_name", REQUIRED_VARS)
    def test_missing_variable_invalid(self, sample_dataset, var_name):
        bad_ds = sample_dataset.drop_vars(var_name)
        assert bad_ds.sidescan.is_valid() is False


@pytest.mark.unit
class TestValidationMissingAttrs:
    """Test that missing attributes raise ValueError."""

    @pytest.mark.parametrize("attr_name", REQUIRED_ATTRS)
    def test_missing_attribute_raises(self, sample_dataset, attr_name):
        bad_ds = sample_dataset.copy()
        del bad_ds.attrs[attr_name]
        with pytest.raises(ValueError, match="Missing attributes"):
            bad_ds.sidescan.validate()

    @pytest.mark.parametrize("attr_name", REQUIRED_ATTRS)
    def test_missing_attribute_invalid(self, sample_dataset, attr_name):
        bad_ds = sample_dataset.copy()
        del bad_ds.attrs[attr_name]
        assert bad_ds.sidescan.is_valid() is False


@pytest.mark.unit
class TestValidationProcessingLevel:
    """Test that processing_level must be known."""

    @pytest.mark.parametrize("level", KNOWN_PROCESSING_LEVELS)
    def test_valid_processing_level_passes(self, sample_dataset, level):
        ds = sample_dataset.copy()
        ds.attrs["processing_level"] = level
        ds.sidescan.validate()

    def test_invalid_processing_level_raises(self, sample_dataset):
        bad_ds = sample_dataset.copy()
        bad_ds.attrs["processing_level"] = "unknown_level"
        with pytest.raises(ValueError, match="'processing_level' must be one of"):
            bad_ds.sidescan.validate()

    def test_invalid_processing_level_invalid(self, sample_dataset):
        bad_ds = sample_dataset.copy()
        bad_ds.attrs["processing_level"] = "unknown_level"
        assert bad_ds.sidescan.is_valid() is False


@pytest.mark.unit
class TestPropertyProcessingLevel:
    """Test processing_level property."""

    def test_processing_level_returns_value(self, sample_dataset):
        sample_dataset.attrs["processing_level"] = "raw"
        assert sample_dataset.sidescan.processing_level == "raw"

    def test_processing_level_returns_unknown_if_missing(self):
        ds = xr.Dataset()
        assert ds.sidescan.processing_level == "unknown"


@pytest.mark.unit
class TestPropertyProcessingHistory:
    """Test processing_history property."""

    def test_processing_history_parses_json(self, sample_dataset):
        history = sample_dataset.sidescan.processing_history
        assert isinstance(history, list)

    def test_processing_history_empty_on_empty_array(self, sample_dataset):
        sample_dataset.attrs["processing_history"] = "[]"
        history = sample_dataset.sidescan.processing_history
        assert len(history) == 0

    def test_processing_history_returns_list_of_dicts(self, sample_dataset):
        entry = {
            "step": "tvg",
            "timestamp_utc": "2026-05-25T12:00:00",
            "params": {"slope": 30.0},
            "function": "apply_tvg",
            "version": "0.2.0",
        }
        sample_dataset.attrs["processing_history"] = json.dumps([entry])
        history = sample_dataset.sidescan.processing_history
        assert len(history) == 1
        assert history[0]["step"] == "tvg"

    def test_processing_history_returns_empty_on_malformed_json(self):
        ds = xr.Dataset(attrs={"processing_history": "not valid json"})
        history = ds.sidescan.processing_history
        assert history == []

    def test_processing_history_returns_empty_if_missing(self):
        ds = xr.Dataset()
        history = ds.sidescan.processing_history
        assert history == []


@pytest.mark.unit
class TestAppendToHistory:
    """Test append_to_history() method."""

    def test_append_to_history_adds_entry(self, sample_dataset):
        sample_dataset.sidescan.append_to_history(
            step_name="tvg_correction",
            function_name="sidescanpy.processing.radiometric.apply_tvg",
            params={"slope": 30.0, "offset": 20.0},
            version="0.2.0",
        )
        history = sample_dataset.sidescan.processing_history
        assert len(history) == 1
        assert history[0]["step"] == "tvg_correction"

    def test_append_to_history_preserves_prior_entries(self, sample_dataset):
        sample_dataset.sidescan.append_to_history(
            step_name="step1",
            function_name="fn1",
            params={"a": 1},
            version="1.0",
        )
        sample_dataset.sidescan.append_to_history(
            step_name="step2",
            function_name="fn2",
            params={"b": 2},
            version="1.0",
        )
        history = sample_dataset.sidescan.processing_history
        assert len(history) == 2
        assert history[0]["step"] == "step1"
        assert history[1]["step"] == "step2"

    def test_append_to_history_entry_has_required_fields(self, sample_dataset):
        sample_dataset.sidescan.append_to_history(
            step_name="test_step",
            function_name="test_function",
            params={"test": 123},
            version="1.0",
        )
        history = sample_dataset.sidescan.processing_history
        entry = history[0]
        assert "step" in entry
        assert "timestamp_utc" in entry
        assert "params" in entry
        assert "function" in entry
        assert "version" in entry

    def test_append_to_history_timestamp_is_iso_format(self, sample_dataset):
        from datetime import datetime
        sample_dataset.sidescan.append_to_history(
            step_name="test",
            function_name="test_fn",
            params={},
            version="1.0",
        )
        history = sample_dataset.sidescan.processing_history
        timestamp_str = history[0]["timestamp_utc"]
        datetime.fromisoformat(timestamp_str)  # Should not raise



# ============================================================
# FIXTURES
# ============================================================

def _make_time_array(n: int) -> np.ndarray:
    """Build n-element datetime64[ns] array, 1 sec apart."""
    base = np.datetime64("2025-01-01T00:00:00", "ns")
    return base + np.arange(n) * np.timedelta64(1_000_000_000, "ns")


@pytest.fixture
def sample_navigation():
    n = 10
    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), np.linspace(41.0, 41.001, n)),
            "longitude": (("obs",), np.linspace(-71.0, -71.001, n)),
        },
        coords={"time": (("obs",), _make_time_array(n))},
        attrs={"source": "nmea_rmc"},
    )


@pytest.fixture
def sample_attitude():
    n = 10
    return xr.Dataset(
        data_vars={
            "roll_deg":  (("obs",), np.zeros(n, dtype=np.float32)),
            "pitch_deg": (("obs",), np.zeros(n, dtype=np.float32)),
            "heave_m":   (("obs",), np.zeros(n, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n))},
        attrs={"source": "pitch_roll_data_message"},
    )


@pytest.fixture
def sample_dvl():
    n = 10
    return xr.Dataset(
        data_vars={
            "altitude_m": (("obs",), np.full(n, 5.0, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n))},
        attrs={"source": "doppler_velocity_log_data_message"},
    )


@pytest.fixture
def sample_sensor():
    n = 10
    return xr.Dataset(
        data_vars={
            "depth_m": (("obs",), np.linspace(10.0, 12.0, n, dtype=np.float32)),
        },
        coords={"time": (("obs",), _make_time_array(n))},
        attrs={"source": "pressure_sensor_reading_message", "sensor_type": "pressure"},
    )


@pytest.fixture
def empty_navigation():
    return xr.Dataset(
        data_vars={
            "latitude":  (("obs",), np.array([], dtype=np.float64)),
            "longitude": (("obs",), np.array([], dtype=np.float64)),
        },
        coords={"time": (("obs",), np.array([], dtype="datetime64[ns]"))},
        attrs={"source": "nmea_rmc"},
    )


# ============================================================
# CONSTANTS
# ============================================================

@pytest.mark.unit
class TestAuxConstants:
    def test_nav_constants(self):
        assert REQUIRED_NAV_DIMS == {"obs"}
        assert REQUIRED_NAV_COORDS == {"time"}
        assert "latitude" in REQUIRED_NAV_VARS
        assert "longitude" in REQUIRED_NAV_VARS
        assert "source" in REQUIRED_NAV_ATTRS

    def test_att_constants(self):
        assert REQUIRED_ATT_DIMS == {"obs"}
        assert REQUIRED_ATT_COORDS == {"time"}
        assert "roll_deg"  in REQUIRED_ATT_VARS
        assert "pitch_deg" in REQUIRED_ATT_VARS
        assert "heave_m"   in REQUIRED_ATT_VARS
        assert "source"    in REQUIRED_ATT_ATTRS

    def test_dvl_constants(self):
        assert REQUIRED_DVL_DIMS == {"obs"}
        assert REQUIRED_DVL_COORDS == {"time"}
        assert "altitude_m" in REQUIRED_DVL_VARS
        assert "source" in REQUIRED_DVL_ATTRS

    def test_gen_constants(self):
        assert REQUIRED_GEN_DIMS == {"obs"}
        assert REQUIRED_GEN_COORDS == {"time"}
        assert "source" in REQUIRED_GEN_ATTRS
        assert "sensor_type" in REQUIRED_GEN_ATTRS


# ============================================================
# ACCESSOR REGISTRATION
# ============================================================

@pytest.mark.unit
class TestAuxAccessorRegistration:
    def test_navigation_accessor_exists(self, sample_navigation):
        assert hasattr(sample_navigation, "navigation_record")

    def test_attitude_accessor_exists(self, sample_attitude):
        assert hasattr(sample_attitude, "attitude_record")

    def test_dvl_accessor_exists(self, sample_dvl):
        assert hasattr(sample_dvl, "dvl_record")

    def test_sensor_accessor_exists(self, sample_sensor):
        assert hasattr(sample_sensor, "sensor_record")

    def test_navigation_methods_exist(self, sample_navigation):
        for method in ("validate", "is_valid", "is_empty"):
            assert callable(getattr(sample_navigation.navigation_record, method))

    def test_attitude_methods_exist(self, sample_attitude):
        for method in ("validate", "is_valid", "is_empty"):
            assert callable(getattr(sample_attitude.attitude_record, method))

    def test_dvl_methods_exist(self, sample_dvl):
        for method in ("validate", "is_valid", "is_empty"):
            assert callable(getattr(sample_dvl.dvl_record, method))

    def test_sensor_methods_exist(self, sample_sensor):
        for method in ("validate", "is_valid", "is_empty"):
            assert callable(getattr(sample_sensor.sensor_record, method))


# ============================================================
# NAVIGATION
# ============================================================

@pytest.mark.unit
class TestNavigationValid:
    def test_valid_passes(self, sample_navigation):
        sample_navigation.navigation_record.validate()

    def test_valid_is_valid(self, sample_navigation):
        assert sample_navigation.navigation_record.is_valid() is True

    def test_valid_not_empty(self, sample_navigation):
        assert sample_navigation.navigation_record.is_empty() is False


@pytest.mark.unit
class TestNavigationEmpty:
    def test_empty_passes_validation(self, empty_navigation):
        empty_navigation.navigation_record.validate()

    def test_empty_is_empty(self, empty_navigation):
        assert empty_navigation.navigation_record.is_empty() is True


@pytest.mark.unit
class TestNavigationMissing:
    @pytest.mark.parametrize("var", REQUIRED_NAV_VARS)
    def test_missing_var_raises(self, sample_navigation, var):
        bad = sample_navigation.drop_vars(var)
        with pytest.raises(ValueError, match="Navigation"):
            bad.navigation_record.validate()

    @pytest.mark.parametrize("coord", REQUIRED_NAV_COORDS)
    def test_missing_coord_raises(self, sample_navigation, coord):
        bad = sample_navigation.drop_vars(coord)
        with pytest.raises(ValueError, match="Navigation"):
            bad.navigation_record.validate()

    @pytest.mark.parametrize("attr", REQUIRED_NAV_ATTRS)
    def test_missing_attr_raises(self, sample_navigation, attr):
        bad = sample_navigation.copy()
        del bad.attrs[attr]
        with pytest.raises(ValueError, match="Navigation"):
            bad.navigation_record.validate()


@pytest.mark.unit
class TestNavigationTimeDtype:
    def test_wrong_time_dtype_raises(self, sample_navigation):
        bad = sample_navigation.assign_coords(
            time=("obs", np.arange(sample_navigation.sizes["obs"], dtype=np.float64))
        )
        with pytest.raises(ValueError, match="datetime64"):
            bad.navigation_record.validate()


# ============================================================
# ATTITUDE
# ============================================================

@pytest.mark.unit
class TestAttitudeValid:
    def test_valid_passes(self, sample_attitude):
        sample_attitude.attitude_record.validate()

    def test_valid_is_valid(self, sample_attitude):
        assert sample_attitude.attitude_record.is_valid() is True

    def test_valid_not_empty(self, sample_attitude):
        assert sample_attitude.attitude_record.is_empty() is False


@pytest.mark.unit
class TestAttitudeMissing:
    @pytest.mark.parametrize("var", REQUIRED_ATT_VARS)
    def test_missing_var_raises(self, sample_attitude, var):
        bad = sample_attitude.drop_vars(var)
        with pytest.raises(ValueError, match="Attitude"):
            bad.attitude_record.validate()

    @pytest.mark.parametrize("coord", REQUIRED_ATT_COORDS)
    def test_missing_coord_raises(self, sample_attitude, coord):
        bad = sample_attitude.drop_vars(coord)
        with pytest.raises(ValueError, match="Attitude"):
            bad.attitude_record.validate()

    @pytest.mark.parametrize("attr", REQUIRED_ATT_ATTRS)
    def test_missing_attr_raises(self, sample_attitude, attr):
        bad = sample_attitude.copy()
        del bad.attrs[attr]
        with pytest.raises(ValueError, match="Attitude"):
            bad.attitude_record.validate()


@pytest.mark.unit
class TestAttitudeTimeDtype:
    def test_wrong_time_dtype_raises(self, sample_attitude):
        bad = sample_attitude.assign_coords(
            time=("obs", np.arange(sample_attitude.sizes["obs"], dtype=np.float64))
        )
        with pytest.raises(ValueError, match="datetime64"):
            bad.attitude_record.validate()


# ============================================================
# DVL
# ============================================================

@pytest.mark.unit
class TestDVLValid:
    def test_valid_passes(self, sample_dvl):
        sample_dvl.dvl_record.validate()

    def test_valid_is_valid(self, sample_dvl):
        assert sample_dvl.dvl_record.is_valid() is True

    def test_valid_not_empty(self, sample_dvl):
        assert sample_dvl.dvl_record.is_empty() is False


@pytest.mark.unit
class TestDVLMissing:
    @pytest.mark.parametrize("var", REQUIRED_DVL_VARS)
    def test_missing_var_raises(self, sample_dvl, var):
        bad = sample_dvl.drop_vars(var)
        with pytest.raises(ValueError, match="DVL"):
            bad.dvl_record.validate()

    @pytest.mark.parametrize("coord", REQUIRED_DVL_COORDS)
    def test_missing_coord_raises(self, sample_dvl, coord):
        bad = sample_dvl.drop_vars(coord)
        with pytest.raises(ValueError, match="DVL"):
            bad.dvl_record.validate()

    @pytest.mark.parametrize("attr", REQUIRED_DVL_ATTRS)
    def test_missing_attr_raises(self, sample_dvl, attr):
        bad = sample_dvl.copy()
        del bad.attrs[attr]
        with pytest.raises(ValueError, match="DVL"):
            bad.dvl_record.validate()


@pytest.mark.unit
class TestDVLTimeDtype:
    def test_wrong_time_dtype_raises(self, sample_dvl):
        bad = sample_dvl.assign_coords(
            time=("obs", np.arange(sample_dvl.sizes["obs"], dtype=np.float64))
        )
        with pytest.raises(ValueError, match="datetime64"):
            bad.dvl_record.validate()


# ============================================================
# SENSOR (generic)
# ============================================================

@pytest.mark.unit
class TestSensorValid:
    def test_valid_passes(self, sample_sensor):
        sample_sensor.sensor_record.validate()

    def test_valid_is_valid(self, sample_sensor):
        assert sample_sensor.sensor_record.is_valid() is True

    def test_valid_not_empty(self, sample_sensor):
        assert sample_sensor.sensor_record.is_empty() is False


@pytest.mark.unit
class TestSensorMissing:
    @pytest.mark.parametrize("coord", REQUIRED_GEN_COORDS)
    def test_missing_coord_raises(self, sample_sensor, coord):
        bad = sample_sensor.drop_vars(coord)
        with pytest.raises(ValueError, match="Sensor"):
            bad.sensor_record.validate()

    @pytest.mark.parametrize("attr", REQUIRED_GEN_ATTRS)
    def test_missing_attr_raises(self, sample_sensor, attr):
        bad = sample_sensor.copy()
        del bad.attrs[attr]
        with pytest.raises(ValueError, match="Sensor"):
            bad.sensor_record.validate()


@pytest.mark.unit
class TestSensorTimeDtype:
    def test_wrong_time_dtype_raises(self, sample_sensor):
        bad = sample_sensor.assign_coords(
            time=("obs", np.arange(sample_sensor.sizes["obs"], dtype=np.float64))
        )
        with pytest.raises(ValueError, match="Sensor"):
            bad.sensor_record.validate()