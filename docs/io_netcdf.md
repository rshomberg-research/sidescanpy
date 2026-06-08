# NetCDF IO

Read and write individual :class:`SideScanFrequencyChannel` datasets as
standalone `.nc` files. Use this when you want to share or process a
single frequency in isolation without the survey-level wrapper.

For full multi-frequency surveys with auxiliary records, see
[`docs/io_hdf5.md`](io_hdf5.md).

## Entry points

| Function | Use case |
|---|---|
| `write_frequency_channel(channel, path)` | Write a channel to disk. |
| `read_frequency_channel(path)` | Read back as a validated channel. |
| `read_frequency_channel(path, chunks={"ping": N})` | Lazy/dask-backed loading. |

## Conventions

- **Engine.** `h5netcdf`. Standardized across the project.
- **Compression.** Gzip level 4 on `backscatter_raw` and
  `backscatter_intensity`. Per-ping scalars (lat, lon, heading, etc.) are
  uncompressed — overhead exceeds savings on small arrays.
- **Chunking on disk.** `(chunk_ping=256, n_channels=2, n_samples=full)`
  by default. Aligns with the dominant access pattern (per-ping reads).
- **Chunking on read.** Pass `chunks={"ping": N}` to return dask-backed
  arrays. None (default) loads eagerly.
- **String coords.** `channel_id` round-trips as native Python strings
  (`["port", "starboard"]`). The reader normalizes byte-strings if the
  engine returns them.

## Round-trip guarantee

After `write_frequency_channel(ch, path); read_frequency_channel(path)`,
the loaded dataset is `xr.testing.assert_identical` to the original:

- All data values exact (including NaN preservation)
- All coords with original dtype
- All attrs including `processing_history`
- Per-ping fields and dimensions preserved

This is enforced by ~31 unit tests in `tests/unit/io/netcdf/test_round_trip.py`.

## Performance notes

- Eager read of a 50-ping × 2000-sample fixture file: <100 ms typical.
- Lazy read with `chunks={"ping": 256}`: subsequent value access forces
  on-demand decompression of only the requested chunks.
- For large surveys, prefer the HDF5 module — it preserves frequency
  channels and aux records together in one file with the same lazy
  loading semantics.

## Future work

- Verify behavior with non-default chunk sizes on read (e.g.
  `chunks={"ping": 50}` against a 256-ping on-disk chunk size). Currently
  works but emits a "chunks separate the stored chunks" warning.
- Add a multi-channel NetCDF writer that bundles multiple
  `SideScanFrequencyChannel` datasets into one `.nc` with groups (the
  HDF5 writer already does this; the NetCDF version would be a
  convenience).