"""
Unit tests for standard HDF5 writer

2026 rshomberg@uri.edu
"""

import pytest
import h5py
from pathlib import Path

from sidescanpy.io.standard.writer import write_ss_hdf5


@pytest.mark.unit
class TestWriterBasic:
    """Basic write operations"""

    def test_write_creates_file(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        write_ss_hdf5(sample_dataset, output_path)
        assert output_path.exists()

    def test_write_validates_before_write(self, tmp_dir):
        """Invalid dataset should raise before write"""
        import xarray as xr
        bad_ds = xr.Dataset()
        output_path = tmp_dir / "test.ss.h5"
        with pytest.raises(ValueError, match="Missing"):
            write_ss_hdf5(bad_ds, output_path)
        assert not output_path.exists()


@pytest.mark.unit
class TestWriterSchema:
    """Verify HDF5 file structure matches schema"""

    def test_write_has_frequency_group(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        with h5py.File(output_path, "r") as f:
            assert "frequencies" in f
            assert f"hz_{freq_hz}" in f["frequencies"]

    def test_write_has_dataset_inside_frequency_group(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        with h5py.File(output_path, "r") as f:
            freq_group = f["frequencies"][f"hz_{freq_hz}"]
            assert "backscatter_raw" in freq_group
            assert "backscatter_intensity" in freq_group


@pytest.mark.unit
class TestWriterChunking:
    """Verify chunking is applied for dask compatibility"""

    def test_write_chunks_backscatter(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        with h5py.File(output_path, "r") as f:
            ds_path = f["frequencies"][f"hz_{freq_hz}/backscatter_raw"]
            assert ds_path.chunks is not None
            assert ds_path.chunks[0] == 256

    def test_write_applies_compression(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        freq_hz = int(sample_dataset.attrs["freq_hz_start"])
        write_ss_hdf5(sample_dataset, output_path, freq_hz=freq_hz)

        with h5py.File(output_path, "r") as f:
            ds_path = f["frequencies"][f"hz_{freq_hz}/backscatter_raw"]
            assert ds_path.compression == "gzip"


@pytest.mark.unit
class TestWriterFileHandling:
    """Test file existence and overwrite handling"""

    def test_file_exists_raises_without_overwrite(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        write_ss_hdf5(sample_dataset, output_path)
        
        with pytest.raises(FileExistsError):
            write_ss_hdf5(sample_dataset, output_path, overwrite=False)

    def test_file_exists_overwrites_with_flag(self, sample_dataset, tmp_dir):
        output_path = tmp_dir / "test.ss.h5"
        write_ss_hdf5(sample_dataset, output_path)
        
        # Should not raise
        write_ss_hdf5(sample_dataset, output_path, overwrite=True)
        assert output_path.exists()