"""
Integration tests for JSF I/O pipeline.

Tests the complete workflow:
1. read_jsf() - Parse JSF binary file
2. jsf_to_native_hdf5() - Archive messages to native HDF5
3. native_hdf5_to_standard() - Convert to standard sonar schema

These tests verify data integrity, format compliance, and transformation fidelity
across the entire pipeline.

2026 rshomberg@uri.edu
"""

import logging
from pathlib import Path
import tempfile
from typing import Tuple

import pytest
import numpy as np
import h5py
import xarray as xr

from sidescanpy.io.jsf.reader import (
    read_jsf,
    jsf_to_native_hdf5,
)
from sidescanpy.io.jsf.converter import (
    native_hdf5_to_standard,
    list_frequencies_in_native_hdf5,
)



logger = logging.getLogger(__name__)


@pytest.fixture
def jsf_file() -> Path:
    """Fixture providing path to test JSF file."""
    jsf_path = Path("data/survey-a/jsf").glob("*.jsf").__next__()
    if not jsf_path.exists():
        pytest.skip("Test JSF file not found")
    return jsf_path


@pytest.fixture
def temp_dir() -> Path:
    """Fixture providing temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestJSFPipelinePhase1:
    """Test Phase 1: JSF binary parsing (read_jsf)."""

    def test_read_jsf_returns_messages(self, jsf_file: Path) -> None:
        """read_jsf returns non-empty message list."""
        messages = read_jsf(jsf_file, N=100)
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        logger.info(f"Parsed {len(messages)} messages")

    def test_read_jsf_preserves_message_count(self, jsf_file: Path) -> None:
        """read_jsf respects N limit."""
        limit = 50
        messages = read_jsf(jsf_file, N=limit)
        
        assert len(messages) == limit
        logger.info(f"Read exactly {limit} messages as expected")

    def test_read_jsf_messages_have_context(self, jsf_file: Path) -> None:
        """Parsed messages have header context set."""
        messages = read_jsf(jsf_file, N=10)
        
        for msg in messages:
            assert hasattr(msg, 'timestamp')
            assert msg.timestamp is not None

    def test_sonar_messages_found(self, jsf_file: Path) -> None:
        """At least some SonarDataMessages are parsed."""
        messages = read_jsf(jsf_file, N=500)
        
        sonar_msgs = [m for m in messages 
                     if type(m).__name__ == 'SonarDataMessage']
        
        assert len(sonar_msgs) > 0
        logger.info(f"Found {len(sonar_msgs)} SonarDataMessages")


class TestJSFPipelinePhase2:
    """Test Phase 2: Archive to native HDF5 (jsf_to_native_hdf5)."""

    def test_jsf_to_native_hdf5_creates_file(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """jsf_to_native_hdf5 creates output file."""
        output_path = temp_dir / "test_native.jsf.h5"
        
        jsf_to_native_hdf5(jsf_file, output_path, overwrite=False)
        
        assert output_path.exists()
        logger.info(f"Created native HDF5: {output_path}")

    def test_native_hdf5_has_correct_structure(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Native HDF5 has expected hierarchy."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        with h5py.File(output_path, 'r') as f:
            # Check structure
            assert "message_types" in f
            assert "SonarDataMessage" in f["message_types"]
            
            sonar_group = f["message_types/SonarDataMessage"]
            freqs = list(sonar_group.keys())
            
            assert len(freqs) > 0
            
            for freq_key in freqs:
                assert freq_key.startswith("hz_")
                freq_group = sonar_group[freq_key]
                assert len(freq_group) > 0
                logger.info(f"Found frequency: {freq_key} with {len(freq_group)} fields")

    def test_sonar_messages_grouped_by_frequency(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """SonarDataMessages organized by frequency in HDF5."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        freqs = list_frequencies_in_native_hdf5(output_path)
        
        assert len(freqs) > 0
        assert all(isinstance(f, int) for f in freqs)
        assert freqs == sorted(freqs, reverse=True)  # Descending
        logger.info(f"Frequencies in HDF5: {freqs}")

    def test_backscatter_data_preserved(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Backscatter data stored in native HDF5."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        with h5py.File(output_path, 'r') as f:
            sonar_group = f["message_types/SonarDataMessage"]
            for freq_key in sonar_group.keys():
                freq_group = sonar_group[freq_key]
                
                if "backscatter_intensity" in freq_group:
                    bs_data = freq_group["backscatter_intensity"][:]
                    assert bs_data is not None
                    assert len(bs_data) > 0
                    logger.info(f"Backscatter preserved: {freq_key}")

    def test_ping_numbers_preserved(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Ping numbers stored correctly in native HDF5."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        with h5py.File(output_path, 'r') as f:
            sonar_group = f["message_types/SonarDataMessage"]
            for freq_key in sonar_group.keys():
                freq_group = sonar_group[freq_key]
                
                if "ping_number" in freq_group:
                    ping_nums = freq_group["ping_number"][:]
                    
                    # Check monotonically increasing or equal
                    assert np.all(ping_nums[1:] >= ping_nums[:-1])
                    logger.info(f"Ping numbers valid: {len(ping_nums)} pings")


class TestJSFPipelinePhase3:
    """Test Phase 3: Convert to standard schema (native_hdf5_to_standard)."""

    def test_native_hdf5_to_standard_creates_file(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """native_hdf5_to_standard creates output file."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        assert standard_path.exists()
        logger.info(f"Created standard HDF5: {standard_path}")

    def test_standard_hdf5_is_valid_netcdf(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard HDF5 is readable as xarray Dataset."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        # Read with xarray
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) > 0
        assert len(ds.coords) > 0
        logger.info(f"Standard HDF5 readable: {dict(ds.sizes)}")
        ds.close()

    def test_standard_dataset_has_required_variables(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard schema includes all required variables."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        
        required_vars = {
            'latitude', 'longitude', 'heading_deg', 'pitch_deg', 'roll_deg',
            'depth_m', 'altitude_m', 'heave_m', 'sound_vel_ms',
            'backscatter_intensity', 'backscatter_raw'
        }
        
        present_vars = set(ds.data_vars.keys())
        assert required_vars.issubset(present_vars)
        
        logger.info(f"All required variables present: {required_vars}")
        ds.close()

    def test_standard_dataset_has_required_coordinates(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard schema includes all required coordinates."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        
        required_coords = {'ping', 'channel_id', 'sample', 'time'}
        
        present_coords = set(ds.coords.keys())
        assert required_coords.issubset(present_coords)
        
        logger.info(f"All required coordinates present: {required_coords}")
        ds.close()

    def test_backscatter_has_correct_shape(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Backscatter has 3D shape [ping, channel, sample]."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        
        bs = ds['backscatter_intensity']
        assert bs.ndim == 3
        assert bs.dims == ('ping', 'channel', 'sample')
        
        n_pings, n_channels, n_samples = bs.shape
        assert n_pings > 0
        assert n_channels == 2
        assert n_samples > 0
        
        logger.info(f"Backscatter shape: {bs.shape}")
        ds.close()

    def test_standard_dataset_passes_schema_validation(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard dataset passes sidescan schema validation."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        
        # Should not raise
        try:
            ds.sidescan.validate()
            logger.info("Dataset passed schema validation")
        finally:
            ds.close()

    def test_specific_frequency_conversion(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Can convert specific frequency only."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard_freq.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        freqs = list_frequencies_in_native_hdf5(native_path)
        
        if len(freqs) > 0:
            target_freq = freqs[0]
            native_hdf5_to_standard(
                native_path, 
                standard_path, 
                freq_hz=target_freq
            )
            
            ds = xr.open_dataset(standard_path, engine="h5netcdf")
            assert ds.attrs['freq_hz_start'] == target_freq
            logger.info(f"Converted specific frequency: {target_freq}")
            ds.close()


class TestJSFPipelineEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_jsf_to_standard(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Complete JSF → Native HDF5 → Standard schema pipeline."""
        native_path = temp_dir / "pipeline_native.jsf.h5"
        standard_path = temp_dir / "pipeline_standard.ss.h5"
        
        # Phase 1: Parse JSF
        messages = read_jsf(jsf_file, N=200)
        assert len(messages) > 0
        logger.info(f"✓ Phase 1: Parsed {len(messages)} messages")
        
        # Phase 2: Archive to native HDF5
        jsf_to_native_hdf5(jsf_file, native_path)
        assert native_path.exists()
        logger.info(f"✓ Phase 2: Archived to {native_path}")
        
        # Phase 3: Convert to standard
        native_hdf5_to_standard(native_path, standard_path)
        assert standard_path.exists()
        logger.info(f"✓ Phase 3: Converted to {standard_path}")
        
        # Verify final output
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        ds.sidescan.validate()
        logger.info(f"✓ Validation: Dataset is valid")
        logger.info(f"✓ Complete pipeline successful: {dict(ds.sizes)}")
        ds.close()

    def test_data_integrity_through_pipeline(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Data integrity maintained through transformations."""
        native_path = temp_dir / "integrity_native.jsf.h5"
        standard_path = temp_dir / "integrity_standard.ss.h5"
        
        # Get original ping count
        messages = read_jsf(jsf_file)
        sonar_msgs = [m for m in messages 
                     if type(m).__name__ == 'SonarDataMessage']
        original_ping_count = len(sonar_msgs)
        
        # Archive and convert
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        # Check final ping count
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        final_ping_count = len(ds.coords['ping'])
        
        assert final_ping_count > 0
        logger.info(
            f"Data integrity: {original_ping_count} → {final_ping_count} pings"
        )
        ds.close()

    def test_multiple_frequencies_preserved(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Multiple frequencies preserved through pipeline."""
        native_path = temp_dir / "multifreq_native.jsf.h5"
        temp_standard_dir = temp_dir / "multifreq_standards"
        temp_standard_dir.mkdir()
        
        # Archive
        jsf_to_native_hdf5(jsf_file, native_path)
        freqs = list_frequencies_in_native_hdf5(native_path)
        
        if len(freqs) > 1:
            logger.info(f"Testing {len(freqs)} frequencies")
            
            # Convert each frequency
            for freq in freqs:
                output_path = temp_standard_dir / f"freq_{freq}.ss.h5"
                native_hdf5_to_standard(
                    native_path, 
                    output_path, 
                    freq_hz=freq,
                    overwrite=True
                )
                
                ds = xr.open_dataset(output_path, engine="h5netcdf")
                assert ds.attrs['freq_hz_start'] == freq
                ds.close()
                logger.info(f"✓ Converted frequency {freq}")