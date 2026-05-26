@dataclass
class SlantRangeParams:
    ground_range_resolution_m: float = None  # None = match slant range spacing
    ground_range_max_m: float = None         # None = compute from data
    fill_value: float = np.nan               # value for blanked nadir region


def apply_slant_range_correction(
    ds: xr.Dataset,
    params: SlantRangeParams = None
) -> xr.Dataset:
    """
    Convert backscatter from slant-range to ground-range coordinates.

    Each ping is resampled individually using linear interpolation.
    Nadir region (slant_range < altitude) is set to fill_value.

    Parameters
    ----------
    ds : xr.Dataset
        Valid dataset with processing_level == "radiometric".
    params : SlantRangeParams, optional

    Returns
    -------
    xr.Dataset
        Dataset with 'sample' dimension replaced by 'ground_range' dimension.
        Coordinate 'ground_range_m' (n_ground_samples,) added.
        processing_level updated to "geometric".
    """
    ...