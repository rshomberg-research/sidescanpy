"""
Unit tests for standard HDF5 reader

2026 rshomberg@uri.edu
"""

import pytest
import xarray as xr

from sidescanpy.io.standard.reader import read_ss_hdf5
from sidescanpy.io.standard.writer import write_ss_hdf5


@pytest.mark.unit
class TestReaderBasic:
    """Basic read operations"""

    def test_read_opens_file(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        ds_read = read_ss_hdf5(output_path, freq_hz=freq_hz)
        assert isinstance(ds_read, xr.Dataset)

    def test_read_nonexistent_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            read_ss_hdf5(tmp_dir / "nonexistent.ss.h5", freq_hz=1495000)


@pytest.mark.unit
class TestReaderValidation:
    """Validation on read"""

    def test_read_validates_loaded_dataset(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        ds_read = read_ss_hdf5(output_path, freq_hz=freq_hz)
        ds_read.sidescan.validate()  # Should not raise


@pytest.mark.unit
class TestReaderLazyLoading:
    """Verify dask lazy loading"""

    def test_read_returns_dask_arrays(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        ds_read = read_ss_hdf5(output_path, freq_hz=freq_hz, chunks={"ping": 256})

        # backscatter_raw should be a dask array
        assert hasattr(ds_read["backscatter_raw"].data, "dask")


@pytest.mark.integration
class TestRoundTrip:
    """Write → read comparison"""

    def test_round_trip_preserves_data(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])

        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)
        ds_read = read_ss_hdf5(output_path, freq_hz=freq_hz)

        # Compare shapes
        assert ds_read["backscatter_raw"].shape == sample_dataset["backscatter_raw"].shape

    def test_round_trip_preserves_attributes(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])

        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)
        ds_read = read_ss_hdf5(output_path, freq_hz=freq_hz)

        assert ds_read.attrs["freq_hz_start"] == sample_dataset.attrs["freq_hz_start"]
        assert ds_read.attrs["processing_level"] == sample_dataset.attrs["processing_level"]