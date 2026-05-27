"""
Tests to verify port and starboard channels are kept separate throughout the pipeline.

This module tests each stage of the JSF→Standard pipeline to identify where
channel separation is lost (if at all).
"""

# FIXME: reorganize, some of these are unit tests and some are integration

import json
import logging
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)


class TestChannelSeparationInRawData:
    """Verify raw .ss.h5 files have distinct port/starboard data."""

    @pytest.mark.integration
    def test_real_data_has_different_channels(self):
        """Load real converted data and verify channels differ."""
        # FIXME: need stable test files
        ss_file = Path("data/survey-a/sshf/20251015170335.ss.h5")
        
        if not ss_file.exists():
            pytest.skip("Real data file not found")
        
        ds = xr.open_dataset(
            ss_file,
            group="frequencies/hz_1495000",
            engine="h5netcdf"
        )
        
        backscatter = ds["backscatter_intensity"].values  # (ping, channel, sample)
        
        # Extract channels
        port = backscatter[:, 0, :]      # (ping, sample)
        starboard = backscatter[:, 1, :] # (ping, sample)
        
        logger.info(f"Port channel: min={port.min():.4f}, max={port.max():.4f}, mean={np.nanmean(port):.4f}")
        logger.info(f"Starboard channel: min={starboard.min():.4f}, max={starboard.max():.4f}, mean={np.nanmean(starboard):.4f}")
        
        # Check they're not identical
        are_identical = np.allclose(port, starboard, equal_nan=True)
        
        if are_identical:
            logger.error("✗ Port and starboard channels are IDENTICAL!")
            logger.error("This indicates channel separation was lost in conversion.")
        else:
            logger.info("✓ Port and starboard channels are DIFFERENT")
        
        assert not are_identical, "Port and starboard channels should NOT be identical"

    @pytest.mark.integration
    def test_channels_differ_by_more_than_noise(self):
        """Channels should differ significantly, not just by numerical precision."""

        # FIXME: use fixture
        ss_file = Path("data/survey-a/sshf/20251015170335.ss.h5")
        
        if not ss_file.exists():
            pytest.skip("Real data file not found")
        
        ds = xr.open_dataset(
            ss_file,
            group="frequencies/hz_1495000",
            engine="h5netcdf"
        )
        
        backscatter = ds["backscatter_intensity"].values  # (ping, channel, sample)
        port = backscatter[:, 0, :]
        starboard = backscatter[:, 1, :]
        
        # Compute per-ping channel difference
        # Average absolute difference for each ping
        valid_mask = ~(np.isnan(port) | np.isnan(starboard))
        
        differences = []
        for ping_idx in range(port.shape[0]):
            valid_samples = valid_mask[ping_idx]
            if valid_samples.sum() > 0:
                diff = np.mean(np.abs(port[ping_idx, valid_samples] - starboard[ping_idx, valid_samples]))
                differences.append(diff)
        
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        
        logger.info(f"Channel differences:")
        logger.info(f"  Mean: {mean_diff:.6f}")
        logger.info(f"  Median: {median_diff:.6f}")
        logger.info(f"  Min: {np.min(differences):.6f}")
        logger.info(f"  Max: {np.max(differences):.6f}")
        logger.info(f"  Std: {np.std(differences):.6f}")
        
        # Mean difference should be significant (> 0.001)
        assert mean_diff > 0.001, \
            f"Channel difference ({mean_diff:.6f}) is too small - channels may be identical"


class TestChannelSeparationInNativeHDF5:
    """Verify native HDF5 archives have distinct port/starboard data."""

    @pytest.mark.integration
    def test_native_hdf5_has_both_channels(self):
        """Native HDF5 should store both port and starboard from JSF."""
        import h5py
        
        native_file = Path("data/survey-a/sshf/20251015170335.jsf.h5")
        
        if not native_file.exists():
            pytest.skip("Native HDF5 file not found")
        
        with h5py.File(native_file, 'r') as f:
            sonar_group = f["message_types/SonarDataMessage"]
            
            for freq_key in sonar_group.keys():
                freq_group = sonar_group[freq_key]
                
                # Get backscatter data
                if "backscatter_intensity" in freq_group:
                    bs_data = freq_group["backscatter_intensity"][:]
                    
                    # This should be a list of JSON strings (one per message)
                    n_messages = len(bs_data)
                    logger.info(f"Frequency {freq_key}: {n_messages} messages")
                    
                    # Check a few messages for structure
                    for msg_idx in [0, n_messages//2, n_messages-1]:
                        bs_json = bs_data[msg_idx]
                        if isinstance(bs_json, bytes):
                            bs_json = bs_json.decode('utf-8')
                        
                        try:
                            bs_list = json.loads(bs_json)
                            logger.info(f"  Message {msg_idx}: {len(bs_list)} samples")
                        except:
                            logger.warning(f"  Message {msg_idx}: Could not parse JSON")

    @pytest.mark.integration
    def test_real_raw_data_has_different_channels(self):
        """Load real converted data and verify channels differ."""
        ss_file = Path("data/survey-a/sshf/20251015170335.ss.h5")
        
        if not ss_file.exists():
            pytest.skip("Real data file not found")
        
        ds = xr.open_dataset(
            ss_file,
            group="frequencies/hz_1495000",
            engine="h5netcdf"
        )
        
        backscatter = ds["backscatter_raw"].values  # (ping, channel, sample)
        
        # Extract channels
        port = backscatter[:, 0, :]      # (ping, sample)
        starboard = backscatter[:, 1, :] # (ping, sample)
        
        logger.info(f"Port channel: min={port.min():.4f}, max={port.max():.4f}, mean={np.nanmean(port):.4f}")
        logger.info(f"Starboard channel: min={starboard.min():.4f}, max={starboard.max():.4f}, mean={np.nanmean(starboard):.4f}")
        
        # Check they're not identical
        are_identical = np.allclose(port, starboard, equal_nan=True)
        
        if are_identical:
            logger.error("✗ Port and starboard channels are IDENTICAL!")
            logger.error("This indicates channel separation was lost in conversion.")
        else:
            logger.info("✓ Port and starboard channels are DIFFERENT")
        
        assert not are_identical, "Port and starboard channels should NOT be identical"

class TestChannelSeparationInJSFMessages:
    """Verify JSF messages have distinct port/starboard from the start."""

    @pytest.mark.integration
    def test_jsf_messages_have_channel_field(self):
        """Original JSF messages should have a channel identifier."""
        from sidescanpy.io.jsf import read_jsf
        
        jsf_file = Path("data/survey-a/jsf/20251015170335.jsf")
        
        if not jsf_file.exists():
            pytest.skip("JSF file not found")
        
        messages = read_jsf(jsf_file, N=100)
        
        # Filter sonar messages
        sonar_msgs = [m for m in messages if type(m).__name__ == 'SonarDataMessage']
        
        logger.info(f"Found {len(sonar_msgs)} sonar messages")
        
        # Check for channel field
        channels_found = set()
        for msg in sonar_msgs[:20]:
            if hasattr(msg, 'channel'):
                channels_found.add(msg.channel)
                logger.info(f"  Message has channel: {msg.channel}")
            else:
                logger.warning(f"  Message missing 'channel' field!")
        
        logger.info(f"Unique channels in first 20 messages: {channels_found}")
        
        # Should have at least 2 channels
        assert len(channels_found) >= 2, \
            f"Expected at least 2 channels in JSF messages, found: {channels_found}"

    @pytest.mark.integration
    def test_jsf_messages_channels_differ(self):
        """Sonar messages with same ping number but different channels should have different data."""
        from sidescanpy.io.jsf import read_jsf
        
        jsf_file = Path("data/survey-a/jsf/20251015170335.jsf")
        
        if not jsf_file.exists():
            pytest.skip("JSF file not found")
        
        messages = read_jsf(jsf_file, N=500)
        
        # Filter sonar messages
        sonar_msgs = [m for m in messages if type(m).__name__ == 'SonarDataMessage']
        
        # Group by ping number
        from collections import defaultdict
        pings = defaultdict(lambda: {0: None, 1: None})
        
        for msg in sonar_msgs:
            if hasattr(msg, 'ping_number') and hasattr(msg, 'channel'):
                pings[msg.ping_number][msg.channel] = msg
        
        logger.info(f"Found {len(pings)} unique ping numbers")
        
        # Find pings with both channels
        dual_channel_pings = {
            ping_num: channels for ping_num, channels in pings.items()
            if channels[0] is not None and channels[1] is not None
        }
        
        logger.info(f"Pings with both channels: {len(dual_channel_pings)}")
        
        if len(dual_channel_pings) == 0:
            pytest.skip("No pings with both channels in test data")
        
        # Compare data from same ping, different channels
        for ping_num, channels in list(dual_channel_pings.items())[:5]:
            msg_ch0 = channels[0]
            msg_ch1 = channels[1]
            
            if hasattr(msg_ch0, 'samples') and hasattr(msg_ch1, 'samples'):
                samples_ch0 = np.array(msg_ch0.samples)
                samples_ch1 = np.array(msg_ch1.samples)
                
                # They might have different lengths
                min_len = min(len(samples_ch0), len(samples_ch1))
                
                are_same = np.allclose(samples_ch0[:min_len], samples_ch1[:min_len])
                
                logger.info(f"Ping {ping_num}:")
                logger.info(f"  Ch0 samples: min={samples_ch0.min()}, max={samples_ch0.max()}, mean={samples_ch0.mean():.4f}")
                logger.info(f"  Ch1 samples: min={samples_ch1.min()}, max={samples_ch1.max()}, mean={samples_ch1.mean():.4f}")
                logger.info(f"  Channels identical: {are_same}")
                
                assert not are_same, \
                    f"Ping {ping_num}: Channel 0 and Channel 1 should be different, but are identical!"


class TestChannelSeparationInConverter:
    """Test that native HDF5 → standard conversion preserves channel separation."""

    @pytest.mark.integration
    def test_converter_preserves_channel_differences(self):
        """After conversion to standard schema, channels should still differ."""
        native_file = Path("data/survey-a/sshf/20251015170335.jsf.h5")
        standard_file = Path("data/survey-a/sshf/20251015170335.ss.h5")
        
        if not native_file.exists():
            pytest.skip("Native HDF5 file not found")
        
        if not standard_file.exists():
            pytest.skip("Standard HDF5 file not found")
        
        # Load standard data
        ds = xr.open_dataset(
            standard_file,
            group="frequencies/hz_1495000",
            engine="h5netcdf"
        )
        
        backscatter = ds["backscatter_intensity"].values  # (ping, channel, sample)
        port = backscatter[:, 0, :]
        starboard = backscatter[:, 1, :]
        
        # Compute per-ping statistics
        valid_mask = ~(np.isnan(port) | np.isnan(starboard))
        
        differences = []
        for ping_idx in range(min(100, port.shape[0])):  # First 100 pings
            valid_samples = valid_mask[ping_idx]
            if valid_samples.sum() > 10:  # Need at least 10 samples
                diff = np.mean(np.abs(port[ping_idx, valid_samples] - starboard[ping_idx, valid_samples]))
                differences.append(diff)
        
        if not differences:
            pytest.skip("Not enough valid data to test")
        
        mean_diff = np.mean(differences)
        
        logger.info(f"Converter test:")
        logger.info(f"  Mean channel difference: {mean_diff:.6f}")
        logger.info(f"  Sample diffs: {differences[:5]}")
        
        assert mean_diff > 0.0001, \
            f"After conversion, channels should differ. Mean difference: {mean_diff}"