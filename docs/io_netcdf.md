# NetCDF

`sidescanpy.io.netcdf` writes one frequency channel on its own, as a
standalone NetCDF-4 file (`.nc`), without the survey wrapper or the aux
records. Use it to hand one frequency to a tool or a colleague that
expects a single dataset. For a whole survey, use [`.ss.h5`](io_hdf5.md).

```python
from sidescanpy.io.netcdf import read_frequency_channel, write_frequency_channel

channel = survey.get_frequency("hz_400000")
write_frequency_channel(channel, "line_001_400khz.nc")
channel = read_frequency_channel("line_001_400khz.nc")
channel = read_frequency_channel("line_001_400khz.nc", chunks={"ping": 256})  # lazy
```

The file holds the same variables and attributes as one
`frequencies/hz_<Hz>` group of an `.ss.h5` file, `processing_history`
included, with the same encoding: gzip 4 with shuffle on variables with
two or more dimensions, chunks of 256 pings, float64 stored as float32
except latitude and longitude. It reads back identical to what was
written, apart from that float32 rounding.

`xr.open_dataset("line_001_400khz.nc", engine="h5netcdf")` opens it
without sidescanpy.
