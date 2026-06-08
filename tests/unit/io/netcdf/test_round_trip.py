"""
Unit tests for sidescanpy.io.netcdf reader/writer.

NetCDF module handles single SideScanFrequencyChannel datasets — no aux records,
no group structure. Used for sharing or processing one frequency in isolation.

2026 rshomberg@uri.edu
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sidescanpy.core.survey import SideScanFrequencyChannel
from sidescanpy.io.netcdf.writer import write_frequency_channel
from sidescanpy.io.netcdf.reader import read_frequency_channel


# ============================================================
# WRITE
# ============================================================

@pytest.mark.unit
class TestWrite:
    def test_creates_file(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_accepts_path_object(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        assert p.exists()

    def test_accepts_string_path(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, str(p))
        assert p.exists()

    def test_overwrite_false_raises_when_exists(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        with pytest.raises(FileExistsError):
            write_frequency_channel(sample_survey_channel, p)

    def test_overwrite_true_replaces(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        original_size = p.stat().st_size
        # Re-write the same data; should succeed and produce a similar-size file
        write_frequency_channel(sample_survey_channel, p, overwrite=True)
        assert p.exists()
        # Size should be roughly comparable (allow 50% wiggle for HDF5 metadata jitter)
        assert 0.5 * original_size <= p.stat().st_size <= 1.5 * original_size

    def test_overwrite_default_is_false(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        # Default (no overwrite kwarg) must raise
        with pytest.raises(FileExistsError):
            write_frequency_channel(sample_survey_channel, p)


# ============================================================
# READ
# ============================================================

@pytest.mark.unit
class TestRead:
    def test_returns_frequency_channel(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert isinstance(loaded, SideScanFrequencyChannel)

    def test_loaded_dataset_validates(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        loaded.dataset.sidescan.validate()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_frequency_channel(tmp_path / "does_not_exist.nc")

    def test_freq_key_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert loaded.freq_key == sample_survey_channel.freq_key


# ============================================================
# ROUND-TRIP IDENTITY (the load-bearing tests)
# ============================================================

@pytest.mark.unit
class TestRoundTripIdentity:
    def test_dataset_identical(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        xr.testing.assert_identical(loaded.dataset, sample_survey_channel.dataset)

    def test_dimensions_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert loaded.n_pings    == sample_survey_channel.n_pings
        assert loaded.n_channels == sample_survey_channel.n_channels
        assert loaded.n_samples  == sample_survey_channel.n_samples

    def test_attrs_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        for key in (
            "freq_hz_start", "freq_hz_end",
            "pulse_length_us", "sample_rate_hz",
            "processing_level", "processing_history",
        ):
            assert loaded.dataset.attrs[key] == sample_survey_channel.dataset.attrs[key]

    def test_processing_history_parses_after_load(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        history = loaded.dataset.sidescan.processing_history
        assert isinstance(history, list)


# ============================================================
# DTYPE & COORDINATE ROUND-TRIPS (the gotcha tests)
# ============================================================

@pytest.mark.unit
class TestDtypePreservation:
    def test_time_coord_is_datetime64_ns(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert loaded.dataset["time"].dtype == np.dtype("datetime64[ns]")

    def test_time_values_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        np.testing.assert_array_equal(
            loaded.dataset["time"].values,
            sample_survey_channel.dataset["time"].values,
        )

    def test_ping_number_dtype_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert loaded.dataset["ping_number"].dtype == np.int32

    def test_channel_id_returns_strings_not_bytes(self, sample_survey_channel, tmp_path):
        """h5netcdf can return strings as bytes depending on encoding flags.
        Our reader must always produce native Python str / numpy.str_."""
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        ids = list(loaded.dataset.channel_id.values)
        assert ids == ["port", "starboard"]
        for v in ids:
            assert isinstance(v, str), f"channel_id element {v!r} is {type(v).__name__}, expected str"

    def test_channel_id_selectable_by_name(self, sample_survey_channel, tmp_path):
        """The whole point of the channel_id coord is name-based selection.
        Verify .sel(channel_id="port") works after round-trip."""
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        port = loaded.dataset["backscatter_intensity"].sel(channel_id="port")
        stbd = loaded.dataset["backscatter_intensity"].sel(channel_id="starboard")
        assert port.shape == (loaded.n_pings, loaded.n_samples)
        assert stbd.shape == (loaded.n_pings, loaded.n_samples)

    def test_backscatter_dtype_preserved(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        assert loaded.dataset["backscatter_raw"].dtype       == np.float32
        assert loaded.dataset["backscatter_intensity"].dtype == np.float32


# ============================================================
# NUMERICAL VALUE PRESERVATION
# ============================================================

@pytest.mark.unit
class TestValuePreservation:
    def test_backscatter_values_exact(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        np.testing.assert_array_equal(
            loaded.dataset["backscatter_raw"].values,
            sample_survey_channel.dataset["backscatter_raw"].values,
        )
        np.testing.assert_array_equal(
            loaded.dataset["backscatter_intensity"].values,
            sample_survey_channel.dataset["backscatter_intensity"].values,
        )

    def test_per_ping_scalars_exact(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        for var in (
            "latitude", "longitude", "heading_deg", "depth_m",
            "roll_deg", "pitch_deg", "heave_m", "altitude_m", "sound_vel_ms",
        ):
            np.testing.assert_array_equal(
                loaded.dataset[var].values,
                sample_survey_channel.dataset[var].values,
            )

    def test_ping_number_values_exact(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        np.testing.assert_array_equal(
            loaded.dataset["ping_number"].values,
            sample_survey_channel.dataset["ping_number"].values,
        )

    def test_nan_values_preserved(self, sample_dataset, tmp_path):
        """heave_m is NaN-filled by the converter. NaN must round-trip."""
        # heave_m in sample_dataset is np.zeros(...). Force NaN explicitly.
        ds = sample_dataset.copy(deep=True)
        ds["heave_m"][:] = np.nan
        ch = SideScanFrequencyChannel(ds)

        p = tmp_path / "channel.nc"
        write_frequency_channel(ch, p)
        loaded = read_frequency_channel(p)
        assert np.all(np.isnan(loaded.dataset["heave_m"].values))


# ============================================================
# LAZY LOADING (chunks parameter)
# ============================================================

@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks")
class TestLazyLoading:
    def test_default_load_is_eager(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p)
        # Eager: backscatter is a numpy array (no dask wrapping)
        backscatter = loaded.dataset["backscatter_raw"].variable.data
        assert isinstance(backscatter, np.ndarray)

    def test_chunks_param_returns_dask(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p, chunks={"ping": 5})

        import dask.array as da
        backscatter = loaded.dataset["backscatter_raw"].variable.data
        assert isinstance(backscatter, da.Array)

    def test_chunks_match_request(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        loaded = read_frequency_channel(p, chunks={"ping": 5})
        # Default sample_dataset is 10 pings; chunks={"ping": 5} → 2 chunks of 5
        chunks_along_ping = loaded.dataset["backscatter_raw"].variable.data.chunks[0]
        assert chunks_along_ping == (5, 5)

    def test_lazy_load_data_equals_eager(self, sample_survey_channel, tmp_path):
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)
        eager = read_frequency_channel(p)
        lazy  = read_frequency_channel(p, chunks={"ping": 5})
        np.testing.assert_array_equal(
            lazy.dataset["backscatter_raw"].values,   # forces compute
            eager.dataset["backscatter_raw"].values,
        )

    def test_aligned_chunks_no_warning(self, tmp_path):
        """Chunks aligned to on-disk layout should not warn."""
        from tests.conftest import make_minimal_sidescan_dataset
        # Use a fixture large enough for default 256-ping on-disk chunks to fit
        ds = make_minimal_sidescan_dataset(n_pings=512, n_samples=100)
        ch = SideScanFrequencyChannel(ds)
        p = tmp_path / "aligned.nc"
        write_frequency_channel(ch, p)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            loaded = read_frequency_channel(p, chunks={"ping": 256})
            # Force evaluation
            _ = loaded.dataset["backscatter_raw"].values


# ============================================================
# COMPRESSION
# ============================================================

@pytest.mark.unit
class TestCompression:
    def test_backscatter_is_compressed(self, sample_survey_channel, tmp_path):
        """backscatter_raw and backscatter_intensity should have gzip applied."""
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)

        # Inspect the file with h5netcdf (read encoding off the file directly)
        import h5netcdf
        with h5netcdf.File(str(p), "r") as f:
            for var_name in ("backscatter_raw", "backscatter_intensity"):
                v = f[var_name]
                # gzip compression is reported via filters / compression attribute
                assert v.compression == "gzip", (
                    f"{var_name} should be gzip-compressed, got {v.compression}"
                )

    def test_per_ping_scalars_not_compressed(self, sample_survey_channel, tmp_path):
        """Compression on small per-ping arrays is overhead. Only big arrays
        should be compressed."""
        p = tmp_path / "channel.nc"
        write_frequency_channel(sample_survey_channel, p)

        import h5netcdf
        with h5netcdf.File(str(p), "r") as f:
            for var_name in ("latitude", "longitude", "heading_deg"):
                v = f[var_name]
                assert v.compression is None, (
                    f"{var_name} should not be compressed, got {v.compression}"
                )

    def test_compressed_file_smaller_than_raw_payload(self, tmp_path):
        """Sanity: compression actually reduces file size for non-random data.

        Uses a constant-valued backscatter dataset (highly compressible) so
        the compressed file is meaningfully smaller than the raw payload size.
        """
        from tests.conftest import make_minimal_sidescan_dataset
        ds = make_minimal_sidescan_dataset(
            n_pings=50, n_samples=2000, backscatter_val=1000.0,
        )
        ch = SideScanFrequencyChannel(ds)
        p = tmp_path / "compressed.nc"
        write_frequency_channel(ch, p)

        # Raw payload: 50 * 2 * 2000 * 4 bytes (float32) * 2 vars (raw + intensity)
        raw_bytes = 50 * 2 * 2000 * 4 * 2
        # File should be substantially smaller than raw because the constant
        # backscatter compresses to almost nothing
        assert p.stat().st_size < raw_bytes * 0.5