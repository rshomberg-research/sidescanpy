'''
Implements core sidescan dataset functionality

2025 rshomberg@uri.edu'''

from dataclasses import dataclass
import xarray as xr
from pathlib import Path
from typing import ClassVar

# TODO: implement


@dataclass(frozen=True)
class SidescanDataset:
    """
    Read-only representation of a standardized sidescan sonar dataset.
    """
    _ds: xr.Dataset

    # Define expected variables (optional schema)
    REQUIRED_VARS: ClassVar[list[str]] = [
        "intensity", "altitude", "roll", "pitch", "heading", "time"
    ]
    REQUIRED_DIMS: ClassVar[list[str]] = ["ping", "sample"]

    # -------------------------------------------------------------------------
    # Class construction
    # -------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "SidescanDataset":
        """Load from a NetCDF file and validate."""
        ds = xr.load_dataset(path)
        cls._validate(ds)
        return cls(ds)
    
    @classmethod
    def load_dir(cls, 
                 path: str | Path, 
                 chunk_sz: int = 1024
                 ) -> "SidescanDataset":
        '''Load from directory of NETCDF files and validate'''
        nc_files = sorted(Path(path).glob('*.nc'))
        ds = xr.open_mfdataset(
            nc_files,
            chunks={'ping': chunk_sz, 'sample': None},
            combine='by_coords'
            )
        cls._validate(ds)
        return cls(ds)

    def save(self, path: str | Path) -> None:
        """Save to NetCDF (non-editable internal dataset)."""
        self._ds.to_netcdf(path)

    # -------------------------------------------------------------------------
    # Data access (read-only)
    # -------------------------------------------------------------------------
    @property
    def ds(self) -> xr.Dataset:
        """Return the underlying xarray dataset (read-only)."""
        return self._ds.copy(deep=False)  # prevent inplace edits

    # convenience properties
    @property
    def intensity(self):
        return self._ds["intensity"]

    @property
    def coords(self):
        return self._ds.coords

    # -------------------------------------------------------------------------
    # Validation logic
    # -------------------------------------------------------------------------
    @classmethod
    def _validate(cls, ds: xr.Dataset):
        missing_vars = [v for v in cls.REQUIRED_VARS if v not in ds]
        missing_dims = [d for d in cls.REQUIRED_DIMS if d not in ds.dims]
        if missing_vars or missing_dims:
            raise ValueError(
                f"Invalid dataset. Missing vars={missing_vars}, dims={missing_dims}"
            )


if __name__=='__main__':
    
    dir = Path('data/survey/sidescan/refined')
    
    ds = SidescanDataset.load_dir(dir)
    print(ds)