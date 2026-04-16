# FIXME: this is not used, but I think it should be

import numpy as np

def calculate_dvl_altitude_plane_fit(beam_distances, beam_angle=20.0):
    """
    Calculate altitude, pitch, and roll from DVL 4-beam distances using plane fitting.
    
    This method fits a plane to the 4 beam endpoints and extracts:
    - Altitude (vertical distance to the fitted plane)
    - Pitch (rotation about cross-track axis)
    - Roll (rotation about along-track axis)
    
    DVL beams are typically arranged in a Janus configuration:
    - Beam 1: Forward-starboard (+x, +y)
    - Beam 2: Aft-starboard (-x, +y)
    - Beam 3: Aft-port (-x, -y)
    - Beam 4: Forward-port (+x, -y)
    
    Parameters:
    -----------
    beam_distances : array-like
        Bottom distances for 4 beams, shape (n_samples, 4) or (4,)
    beam_angle : float
        Beam angle from vertical in degrees (default: 20.0)
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'altitude': Vertical distance to bottom plane (m)
        - 'pitch': Pitch angle from plane fit (degrees, positive = bow down)
        - 'roll': Roll angle from plane fit (degrees, positive = starboard down)
        - 'altitude_std': RMS residual of plane fit (m)
        - 'altitude_per_beam': Vertical distance for each beam (m)
        - 'bottom_points': 3D coordinates of beam endpoints (m)
        - 'plane_normal': Unit normal vector of fitted plane
        - 'quality_flag': 0=good, 1=suspect, 2=bad (insufficient beams)
    """
    
    beam_distances = np.asarray(beam_distances)
    beam_angle_rad = np.radians(beam_angle)
    
    # Handle both single measurement and time series
    single_measurement = beam_distances.ndim == 1
    if single_measurement:
        beam_distances = beam_distances.reshape(1, -1)
    
    n_samples = beam_distances.shape[0]
    
    # Define beam unit vectors in DVL coordinate frame
    # DVL frame: x=forward, y=starboard, z=down
    sin_a = np.sin(beam_angle_rad)
    cos_a = np.cos(beam_angle_rad)
    
    # Beam unit vectors (normalized)
    beam_vectors = np.array([
        [sin_a, sin_a, cos_a],    # Beam 1: forward-starboard
        [-sin_a, sin_a, cos_a],   # Beam 2: aft-starboard
        [-sin_a, -sin_a, cos_a],  # Beam 3: aft-port
        [sin_a, -sin_a, cos_a]    # Beam 4: forward-port
    ])
    
    # Initialize output arrays
    altitude = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    roll = np.zeros(n_samples)
    altitude_std = np.zeros(n_samples)
    quality_flag = np.zeros(n_samples, dtype=int)
    plane_normals = np.zeros((n_samples, 3))
    all_bottom_points = np.zeros((n_samples, 4, 3))
    altitude_per_beam = np.zeros((n_samples, 4))
    
    for i in range(n_samples):
        # Get valid beams
        valid = ~np.isnan(beam_distances[i]) & (beam_distances[i] > 0)
        n_valid = np.sum(valid)
        
        if n_valid < 3:
            # Need at least 3 points to fit a plane
            quality_flag[i] = 2
            altitude[i] = np.nan
            pitch[i] = np.nan
            roll[i] = np.nan
            continue
        
        # Calculate 3D positions of beam endpoints
        # bottom_point = beam_distance * beam_unit_vector
        bottom_points = beam_distances[i, valid, np.newaxis] * beam_vectors[valid]
        all_bottom_points[i, valid] = bottom_points
        
        # Fit plane to bottom points using least squares
        # Plane equation: ax + by + cz + d = 0
        # Normalized form: nx*x + ny*y + nz*z = d (where [nx,ny,nz] is unit normal)
        
        if n_valid == 3:
            # Exactly 3 points - compute plane directly
            p1, p2, p3 = bottom_points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
        else:
            # 4 points - use least squares plane fitting
            # Fit plane by finding normal vector that minimizes squared distances
            centroid = np.mean(bottom_points, axis=0)
            centered = bottom_points - centroid
            
            # SVD to find best-fit plane
            _, _, vh = np.linalg.svd(centered)
            normal = vh[-1]  # Last row = normal to best-fit plane
        
        # Ensure normal points upward (negative z in DVL frame)
        if normal[2] > 0:
            normal = -normal
        
        plane_normals[i] = normal
        
        # Calculate altitude as distance from origin to plane along z-axis
        # Distance from origin to plane: d = normal · point_on_plane
        d = np.dot(normal, bottom_points[0])
        altitude[i] = d / abs(normal[2])  # Project onto vertical
        
        # Calculate pitch and roll from plane normal
        # Normal vector: [nx, ny, nz] where nz ≈ -1 for level bottom
        # Pitch: rotation about y-axis (cross-track)
        # Roll: rotation about x-axis (along-track)
        
        nx, ny, nz = normal
        
        # pitch = arctan(nx / -nz)  # positive when bow down
        # roll = arctan(ny / -nz)   # positive when starboard down
        pitch[i] = np.degrees(np.arctan2(nx, -nz))
        roll[i] = np.degrees(np.arctan2(ny, -nz))
        
        # Calculate residuals (how well points fit the plane)
        distances_to_plane = np.abs(np.dot(bottom_points, normal) - d)
        altitude_std[i] = np.sqrt(np.mean(distances_to_plane**2))
        
        # Altitude per beam (vertical component only)
        altitude_per_beam[i, valid] = bottom_points[:, 2]
        
        # Quality flag
        if altitude_std[i] > 0.5:
            quality_flag[i] = 1  # Suspect - high residuals
        elif n_valid < 4:
            quality_flag[i] = 1  # Suspect - missing beam
    
    # Prepare results
    results = {
        'altitude': altitude,
        'pitch': pitch,
        'roll': roll,
        'altitude_std': altitude_std,
        'altitude_per_beam': altitude_per_beam,
        'bottom_points': all_bottom_points,
        'plane_normal': plane_normals,
        'quality_flag': quality_flag,
        'n_valid_beams': np.sum(~np.isnan(beam_distances) & (beam_distances > 0), axis=1)
    }
    
    # If single measurement, return scalars
    if single_measurement:
        for key in ['altitude', 'pitch', 'roll', 'altitude_std', 'n_valid_beams']:
            results[key] = results[key][0]
        results['quality_flag'] = results['quality_flag'][0]
        results['altitude_per_beam'] = results['altitude_per_beam'][0]
        results['bottom_points'] = results['bottom_points'][0]
        results['plane_normal'] = results['plane_normal'][0]
    
    return results
