
import numpy as np

def estimate_attitude_from_dvl_beams(   # TODO: test/refactor (chatGPT)
    ranges,
    beam_angle_deg=20.0,
    beam_azimuths_deg=None,
    lever_arm_body=None,
    valid_mask=None,
    return_details=False
    ):
    """
    Estimate altitude (down positive), pitch, and roll from N beam ranges by fitting a plane.

    Parameters
    ----------
    ranges : array_like, shape (N,)
        Beam ranges (meters) measured from instrument to seabed (slant ranges).
    beam_angle_deg : float
        Inclination of the beams from vertical (0 = down). For DVL '20 deg' means 20° from vertical.
    beam_azimuths_deg : array_like, shape (N,), optional
        Azimuths of each beam in body frame, degrees. Default: [45,135,225,315] for 4-beam Janus.
        Azimuth measured from forward (x) toward starboard (y): 0° = forward, 90° = starboard.
    lever_arm_body : array_like (3,), optional
        Vector (x_forward, y_starboard, z_down) from vehicle reference to instrument phase center.
        If provided, points will be offset by this before fitting.
    valid_mask : array_like bool (N,), optional
        Mask indicating valid beams; invalid beams are ignored.
    return_details : bool
        If True, returns (altitude, pitch_deg, roll_deg, plane_coeffs, residuals, points_body)

    Returns
    -------
    altitude : float
        Vertical distance from instrument to seabed (down-positive), meters.
    pitch_deg : float
        Estimated pitch in degrees (nose-up positive). Signed accordingly.
    roll_deg : float
        Estimated roll in degrees (right-wing-down positive).
    (plus optional details)
    """

    ranges = np.asarray(ranges, dtype=float).ravel()
    N = ranges.size
    if beam_azimuths_deg is None:
        # default to Janus 4-beam diagonal config
        if N == 4:
            beam_azimuths_deg = np.array([45.0, 135.0, 225.0, 315.0])
        else:
            # default equally spaced azimuths around circle
            beam_azimuths_deg = np.linspace(0, 360, N, endpoint=False)

    beam_azimuths_deg = np.asarray(beam_azimuths_deg, dtype=float).ravel()
    if beam_azimuths_deg.size != N:
        raise ValueError("beam_azimuths_deg must have same length as ranges")

    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("No valid beams")

    # Beam unit vectors in body frame (x forward, y starboard, z down)
    a = np.deg2rad(beam_angle_deg)   # inclination from vertical
    phis = np.deg2rad(beam_azimuths_deg)
    sin_a = np.sin(a); cos_a = np.cos(a)
    bx = sin_a * np.cos(phis)
    by = sin_a * np.sin(phis)
    bz = cos_a * np.ones_like(phis)
    B = np.vstack([bx, by, bz]).T   # shape (N,3)

    # Points in body frame: p = r * b
    p_body = (ranges[:, None] * B)   # (N,3)

    # Apply lever-arm (if lever_arm given is from nav->instrument in body frame,
    # subtract to get seabed point relative to nav origin; here we want points relative to instrument origin)
    if lever_arm_body is not None:
        lever = np.asarray(lever_arm_body, dtype=float).reshape(3,)
        # If lever is from vehicle origin to instrument, subtract it: points relative to vehicle origin = p_body + lever
        p_body = p_body + lever[None, :]

    # Select valid beams
    p = p_body[valid_mask, :]  # m x 3

    # Fit plane z = a*x + b*y + c  by least squares
    # Build design matrix: [x, y, 1] * [a, b, c] = z
    X = np.column_stack([p[:, 0], p[:, 1], np.ones(p.shape[0])])
    z = p[:, 2]

    # Solve in least squares sense
    # If underdetermined (e.g., <3 points), np.linalg.lstsq will still return something but check rank
    coeffs, *_ = np.linalg.lstsq(X, z, rcond=None)  # coeffs = [a, b, c]
    a_coeff, b_coeff, c_coeff = coeffs

    # Residuals (per-beam)
    z_fit = X @ coeffs
    residuals = z - z_fit

    # Compute pitch and roll from plane slopes
    # For plane z = a x + b y + c, slope in x is a, slope in y is b.
    # If seabed is flat and plane tilt is caused by vehicle pitch/roll, then:
    # pitch ≈ -atan(a), roll ≈ -atan(b)  (signs depend on axis conventions)
    pitch_rad = -np.arctan(a_coeff)
    roll_rad = -np.arctan(b_coeff)

    # altitude is c_coeff (z at x=0,y=0) -> down-positive
    altitude = float(c_coeff)

    # Convert to degrees and adopt sign conventions:
    pitch_deg = np.rad2deg(pitch_rad)   # nose-up positive (if you use that convention)
    roll_deg = np.rad2deg(roll_rad)     # right-wing-down positive

    # if return_details:
    #     return altitude, pitch_deg, roll_deg, (a_coeff, b_coeff, c_coeff), residuals, p_body
    return altitude, pitch_deg, roll_deg
