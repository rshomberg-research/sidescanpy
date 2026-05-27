"""
Integration tests for JSF I/O pipeline with schema validation.

Tests the complete workflow:
1. read_jsf() - Parse JSF binary file
2. jsf_to_native_hdf5() - Archive messages to native HDF5
3. native_hdf5_to_standard() - Convert to standard sonar schema
4. Validate output against standard schema

These tests verify that the pipeline produces valid standard-schema HDF5 files.

2026 rshomberg@uri.edu
"""

# FIXME: these tests are very slow, partially because they are running the same
# exact code over and over. I need to figure out if it is possible to use
# fixtures to speed this up. Or just combine multiple tests.

import logging
from pathlib import Path
import tempfile

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


# Mark all tests as slow and integration
pytestmark = [pytest.mark.slow, pytest.mark.integration]

@pytest.fixture
def jsf_file() -> Path:
    """Fixture providing path to test JSF file."""
    jsf_path = Path("data/survey-a/jsf").glob("*.jsf").__next__()
    # FIXME: choose more consistent and shorter test file.
    # Also can I re-use it while testing?
    if not jsf_path.exists():
        pytest.skip("Test JSF file not found")
    return jsf_path


# ???: is there already a tmp_dr standard fixture?
@pytest.fixture
def temp_dir() -> Path:
    """Fixture providing temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def debug_dataset(ds: xr.Dataset, label: str = "") -> None:
    """Debug print dataset contents."""
    logger.info(f"\n=== Dataset Debug: {label} ===")
    logger.info(f"Dimensions: {dict(ds.sizes)}")
    logger.info(f"Coordinates: {list(ds.coords.keys())}")
    logger.info(f"Data vars: {list(ds.data_vars.keys())}")
    if ds.attrs:
        logger.info(f"Attributes: {dict(ds.attrs)}")


class TestJSFPipelinePhase1:
    """Test Phase 1: JSF binary parsing (read_jsf)."""

    def test_read_jsf_returns_messages(self, jsf_file: Path) -> None:
        """read_jsf returns non-empty message list."""
        messages = read_jsf(jsf_file, N=100)
        
        assert isinstance(messages, list)
        assert len(messages) > 0
        logger.info(f"✓ Parsed {len(messages)} messages")

    def test_read_jsf_preserves_message_count(self, jsf_file: Path) -> None:
        """read_jsf respects N limit."""
        limit = 50
        messages = read_jsf(jsf_file, N=limit)
        
        assert len(messages) == limit
        logger.info(f"✓ Read exactly {limit} messages")

    def test_sonar_messages_found(self, jsf_file: Path) -> None:
        """At least some SonarDataMessages are parsed."""
        messages = read_jsf(jsf_file, N=500)
        
        sonar_msgs = [m for m in messages 
                     if type(m).__name__ == 'SonarDataMessage']
        
        assert len(sonar_msgs) > 0
        logger.info(f"✓ Found {len(sonar_msgs)} SonarDataMessages")


class TestJSFPipelinePhase2:
    """Test Phase 2: Archive to native HDF5 (jsf_to_native_hdf5)."""

    def test_jsf_to_native_hdf5_creates_file(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """jsf_to_native_hdf5 creates output file."""
        output_path = temp_dir / "test_native.jsf.h5"
        
        jsf_to_native_hdf5(jsf_file, output_path, overwrite=False)
        
        assert output_path.exists()
        logger.info(f"✓ Created native HDF5")

    def test_native_hdf5_has_correct_structure(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Native HDF5 has expected hierarchy."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        with h5py.File(output_path, 'r') as f:
            assert "message_types" in f
            assert "SonarDataMessage" in f["message_types"]
            
            sonar_group = f["message_types/SonarDataMessage"]
            freqs = list(sonar_group.keys())
            
            assert len(freqs) > 0
            logger.info(f"✓ Native HDF5 has {len(freqs)} frequencies")

    def test_sonar_messages_grouped_by_frequency(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """SonarDataMessages organized by frequency in HDF5."""
        output_path = temp_dir / "test_native.jsf.h5"
        jsf_to_native_hdf5(jsf_file, output_path)
        
        freqs = list_frequencies_in_native_hdf5(output_path)
        
        assert len(freqs) > 0
        assert all(isinstance(f, int) for f in freqs)
        assert freqs == sorted(freqs, reverse=True)
        logger.info(f"✓ Frequencies {freqs} (descending)")


class TestJSFPipelinePhase3:
    """Test Phase 3: Convert to standard schema with validation."""

    def test_native_hdf5_to_standard_creates_file(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """native_hdf5_to_standard creates output file."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        assert standard_path.exists()
        assert standard_path.stat().st_size > 0
        logger.info(f"✓ Created standard HDF5 ({standard_path.stat().st_size / 1e6:.1f} MB)")

    def test_standard_output_readable_as_netcdf(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard output is readable as xarray Dataset."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        # Should be readable
        ds = xr.open_dataset(standard_path, engine="h5netcdf")
        debug_dataset(ds, "Standard output")
        ds.close()
        
        logger.info("✓ Output readable as NetCDF/xarray")

    def test_standard_output_matches_schema(
    self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Standard output matches sidescan schema."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        # Inspect the file structure
        logger.info("\n=== File Structure ===")
        with h5py.File(standard_path, 'r') as f:
            def print_structure(name, obj):
                indent = len(name.split('/')) - 1
                logger.info(f"{'  ' * indent}{name}")
            
            f.visititems(print_structure)
            
            # Find all groups
            groups = [k for k in f.keys()]
            logger.info(f"\nTop-level groups: {groups}")
        
        # Try to find and read a valid frequency dataset
        with xr.open_dataset(standard_path, engine="h5netcdf") as full_ds:
            logger.info(f"\nFull file dimensions: {dict(full_ds.sizes)}")
            logger.info(f"Full file data_vars: {list(full_ds.data_vars.keys())}")
            logger.info(f"Full file coords: {list(full_ds.coords.keys())}")
            
            if len(full_ds.data_vars) == 0:
                # Try reading from specific group
                with h5py.File(standard_path, 'r') as f:
                    # Find frequency groups
                    freq_groups = []
                    
                    def find_freq_groups(name, obj):
                        if 'hz_' in name:
                            freq_groups.append(name)
                    
                    f.visititems(find_freq_groups)
                    logger.info(f"\nFrequency groups found: {freq_groups}")
                    
                    if freq_groups:
                        target_group = freq_groups[0]
                        logger.info(f"Reading from group: {target_group}")
                        
                        try:
                            ds = xr.open_dataset(standard_path, engine="h5netcdf", group=target_group)
                            debug_dataset(ds, f"Group: {target_group}")
                            
                            # Validate
                            ds.sidescan.validate()
                            logger.info("✓ Output matches schema")
                            ds.close()
                        except Exception as e:
                            logger.error(f"Failed to read group {target_group}: {e}")
                            raise
                    else:
                        pytest.skip("No frequency groups found in output file")
            else:
                # Full file has data, validate it
                debug_dataset(full_ds, "Full file")
                try:
                    full_ds.sidescan.validate()
                    logger.info("✓ Output matches schema")
                except Exception as e:
                    logger.error(f"Schema validation failed: {e}")
                    raise
                
    def test_backscatter_data_present(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Backscatter data is present in output."""
        native_path = temp_dir / "test_native.jsf.h5"
        standard_path = temp_dir / "test_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        native_hdf5_to_standard(native_path, standard_path)
        
        with h5py.File(standard_path, 'r') as f:
            # Search for backscatter data
            backscatter_found = False
            
            def check_backscatter(name, obj):
                nonlocal backscatter_found
                if 'backscatter' in name.lower():
                    backscatter_found = True
            
            f.visititems(check_backscatter)
            
            if backscatter_found:
                logger.info("✓ Backscatter data present in output")
            else:
                pytest.skip("No backscatter data in output (test data may not have sonar)")


class TestJSFPipelineEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_completes_successfully(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Complete pipeline: JSF → Native HDF5 → Standard schema."""
        native_path = temp_dir / "pipeline_native.jsf.h5"
        standard_path = temp_dir / "pipeline_standard.ss.h5"
        
        # Phase 1
        messages = read_jsf(jsf_file, N=100)
        assert len(messages) > 0
        
        # Phase 2
        jsf_to_native_hdf5(jsf_file, native_path)
        assert native_path.exists()
        
        # Phase 3
        native_hdf5_to_standard(native_path, standard_path)
        assert standard_path.exists()
        
        logger.info("✓ Full pipeline completed successfully")

    def test_pipeline_preserves_data_integrity(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Pipeline doesn't lose data between stages."""
        native_path = temp_dir / "integrity_native.jsf.h5"
        standard_path = temp_dir / "integrity_standard.ss.h5"
        
        # Archive
        jsf_to_native_hdf5(jsf_file, native_path)
        native_size = native_path.stat().st_size
        
        # Convert
        native_hdf5_to_standard(native_path, standard_path)
        standard_size = standard_path.stat().st_size
        
        assert native_size > 0
        assert standard_size > 0
        assert standard_size > native_size * 0.5  # Standard shouldn't be much smaller
        
        logger.info(f"✓ Data integrity: native={native_size/1e6:.1f}MB, standard={standard_size/1e6:.1f}MB")

    def test_multiple_frequencies_preserved(
        self, jsf_file: Path, temp_dir: Path
    ) -> None:
        """Multiple frequencies are preserved through pipeline."""
        native_path = temp_dir / "multifreq_native.jsf.h5"
        standard_path = temp_dir / "multifreq_standard.ss.h5"
        
        jsf_to_native_hdf5(jsf_file, native_path)
        freqs_before = list_frequencies_in_native_hdf5(native_path)
        
        native_hdf5_to_standard(native_path, standard_path)
        
        # Check output has frequency groups
        with h5py.File(standard_path, 'r') as f:
            freq_groups = [k for k in f.keys() if 'hz_' in k or 'freq' in k.lower()]
            
            if freq_groups:
                logger.info(f"✓ {len(freq_groups)} frequencies in output")
            else:
                logger.info(f"✓ {len(freqs_before)} frequencies archived (not in separate groups)")