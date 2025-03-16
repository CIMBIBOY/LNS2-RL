import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import evalBSpline

def addObsToGridSeq_DEMO(grid_seq, path):
    '''
        This function adds a 3x3x3 star shaped object to the already generated grid_seq according to the path given. 
        This function should only be used for DEMO purposes, othervise use functionality provided by GridMapEnv class
    '''
    path = path.astype(int)
    occupied_cells_relative = [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]

    for t in range(grid_seq.shape[0]):
        t_path = np.min((t,path.shape[0]-1))
        for coord_add in occupied_cells_relative:
            coord = path[t_path,1:4] + coord_add
            if np.all(np.logical_and(0 <= coord, coord < grid_seq.shape[1:4])):
                grid_seq[tuple([t, *coord])] = 1


def getRandomFreeCell_DEMO(grid_seq, point: tuple = None, r:int|tuple=0, max_attempts:int=100):
    grid = grid_seq[0]

    point = np.array(point)
    if isinstance(r, int): r = np.full(point.shape,r)

    for _ in range(max_attempts):
        if point is None: # Get a random point anywhere on the grid
            point_free = np.random.randint(0,np.array(grid.shape)-1,np.array(grid.shape).shape)
        else: # Get a random point_free in r radius of the given point
            point = np.array(point)
            point_free = np.array([np.random.randint(low=point[i]-r[i], high=point[i]+r[i]) for i in range(point.size)])
            
        if np.all(np.logical_and(0 <= point_free, point_free < grid.shape)): # In array bounds
            if grid[tuple(point_free)] < 0.01: # Not obstacle
                return tuple(point_free)
            
    raise RuntimeError(f"Failed to find a free cell within the grid/ specified radius after {max_attempts} attempts.")

def convertCoord_RealToGrid(coord_real, shape_real, shape_grid):
    ''' ##### This function converts SI distance [km] units to the grid system indexes '''
    coord_real = np.array(coord_real)
    shape_real = np.array(shape_real)
    shape_grid = np.array(shape_grid)
    
    if np.all(np.logical_and(0 <= coord_real, coord_real <= shape_real)): # In array bounds
        coord_grid = np.round(coord_real / shape_real * shape_grid).astype(np.int_)
        coord_grid = np.clip(coord_grid,0,shape_grid-1)
        return coord_grid
    else: 
        print('The provided point is outside the given World coordinate system, please give a point within bounds.')

def convertCoord_GridToReal(coord_grid, shape_real, shape_grid):
    ''' ##### This function converts grid system coordinates to real world SI distance [km] '''
    coord_grid = np.array(coord_grid)
    shape_real = np.array(shape_real)
    shape_grid = np.array(shape_grid)
    
    if np.all(np.logical_and(0 <= coord_grid, coord_grid < shape_grid)): # In array bounds
        coord_grid = np.round(coord_grid / (shape_grid-1) * shape_real).astype(np.int_)
        coord_grid = np.clip(coord_grid,0,shape_real)
        return coord_grid
    else: 
        print('The provided point (index) is greater than (shape_grid - 1), please give a point within bounds.')


def convertWayPointsToSI_DEMO(plan:dict, shape_real, shape_grid, v_knots:float=120, timestep_real=5, debug=False):

    def calcTotalDist(path):
        diff = np.diff(path,axis=0)
        dist = np.linalg.norm(diff, axis=1)
        totaldist = np.sum(dist)
        return totaldist

    # Data conversion
    shape_real = np.array(shape_real)
    shape_grid = np.array(shape_grid)
    v_real_max = v_knots / 1.94
    v_real = v_real_max * 0.9

    # Interpolate an init plan to calculate total distance in grid units
    interp_grid_tcku = plan['path_interp_BSpline_tcku']
    interp_grid = evalBSpline(*interp_grid_tcku, no_points=plan['path_extracted'].shape[0]*2) # Same number of points as path_extracted

    # Get the distance multipliers for distance conversion
    # Optional: Send a warning message if the distance multipliers are really different in each dimension
    mult_dist_grid_to_real_max = np.max(shape_real*1000/(shape_grid-1)) # [m]  max ensures that we dont go over the speed limit
    mult_dist_grid_to_real = shape_real / (shape_grid-1)
    

    # Calculate the total distance in both grid and real
    totaldist_grid = calcTotalDist(interp_grid[:,1:])       # grid cell unit
    totaltime_grid = interp_grid[-1,0] - interp_grid[0,0]   # grid unit

    totaldist_real = totaldist_grid * mult_dist_grid_to_real_max
    totaltime_real= totaldist_real / v_real

    # Interpolate a new spline with a predefined timestep
    no_points = int(totaltime_real/timestep_real)
    interp_real = evalBSpline(*interp_grid_tcku, no_points=no_points, calc_rot=False)
    # Shift the time of the real interpolatiion to start at zero
    interp_real[:,0] = interp_real[:,0] - interp_real[0,0]

    # Scale time and coords to the real coord sys
    scaled_coords = interp_real[:,1:] * mult_dist_grid_to_real * 1000
    mult_time_grid_to_real = totaltime_real / totaltime_grid
    scaled_time = interp_real[:,0] * mult_time_grid_to_real

    interp_real = np.hstack((np.expand_dims(scaled_time,axis=1),scaled_coords))

    totaldist_real_actual = calcTotalDist(interp_real[:,1:])
    totaltime_real_actual = interp_real[-1,0] - interp_real[0,0]


    interp_real_diff = np.diff(interp_real,axis=0)
    interp_real_pos_dist = np.linalg.norm(interp_real_diff[:,1:], axis=1)
    interp_real_vel = interp_real_pos_dist/ interp_real_diff[:,0]
    interp_real_max_vel = np.max(interp_real_vel)

    if debug: 
        print(f'Total distance goal: {totaldist_real/1000:.2f} [km]')
        print(f'Total distance actual: {totaldist_real_actual/1000:.2f} [km]')
        print('')
        print(f'Total time goal: {totaltime_real/60:.2f} [min]')
        print(f'Total time actual: {totaltime_real_actual/60:.2f} [min]')
        print('')
        print(f'Velocity goal: {v_real:.2f} [m/s]')
        print(f'Velocity actual: {totaldist_real_actual/totaltime_real:.2f} [m/s]')
        print(f'Maximum velocity: {interp_real_max_vel:.2f} [m/s]')
        print(f'Maximum velocity permitted: {v_real_max:.2f} [m/s]')

        if False: 
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(interp_real[:,1]/1000, interp_real[:,2]/1000, interp_real[:, 3]/1000, '-')
            ax.set(xlim=(0,shape_real[0]), ylim=(0,shape_real[1]), zlim=(0,shape_real[2]))
            ax.set(xlabel='X [km]', ylabel='Y [km]', zlabel='Z [km]')
            plt.show()

    return interp_real


def convert_xyz_to_latlon(xyz_data, origin_lat, origin_lon, origin_alt):
    """
    Converts xyz coordinates (with an origin in SI units) to latitude, longitude, and elevation.
    
    Args:
        xyz_data: numpy array with time, x, y, z values
        origin_lat: latitude of the origin point
        origin_lon: longitude of the origin point
        origin_alt: altitude (elevation) of the origin point in meters
    
    Returns:
        numpy array with time, latitude, longitude, and elevation (altitude)
    """

    # WGS84 ellipsoid constants
    a = 6378137.0  # Earth's semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f ** 2  # Square of eccentricity

    def geodetic_to_ecef(lat, lon, alt):
        """Convert geodetic coordinates (latitude, longitude, altitude) to ECEF."""
        lat = np.radians(lat)
        lon = np.radians(lon)
        
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - e2) + alt) * np.sin(lat)
        
        return X, Y, Z

    def ecef_to_geodetic(X, Y, Z):
        """Convert ECEF coordinates (X, Y, Z) to geodetic coordinates (lat, lon, alt)."""
        lon = np.arctan2(Y, X)
        
        p = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Z * a, p * (1 - f))
        
        lat = np.arctan2(Z + e2 * (1 - f) * a * np.sin(theta) ** 3,
                        p - e2 * a * np.cos(theta) ** 3)
        
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        
        lat = np.degrees(lat)
        lon = np.degrees(lon)
        
        return lat, lon, alt

    # Convert origin geodetic coordinates to ECEF
    origin_x, origin_y, origin_z = geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    
    # Output array: time, latitude, longitude, elevation
    latlon_data = np.zeros((xyz_data.shape[0], 4))
    
    for i in range(xyz_data.shape[0]):
        time, x, y, z = xyz_data[i]
        
        # Add xyz offset to the origin ECEF coordinates
        new_x = origin_x + x
        new_y = origin_y + y
        new_z = origin_z + z
        
        # Convert the new ECEF coordinates back to geodetic
        lat, lon, alt = ecef_to_geodetic(new_x, new_y, new_z)
        
        latlon_data[i] = [time, lat, lon, alt]
    
    return latlon_data

def convert_latlon_to_xyz(latlon_data, origin_lat, origin_lon, origin_alt):
    """
    Converts latitude, longitude, and elevation (altitude) coordinates to xyz (relative to origin in SI units).

    Args:
        latlon_data: numpy array with time, latitude, longitude, and elevation values
        origin_lat: latitude of the origin point
        origin_lon: longitude of the origin point
        origin_alt: altitude (elevation) of the origin point in meters

    Returns:
        numpy array with time, x, y, z values
    """

    # WGS84 ellipsoid constants
    a = 6378137.0  # Earth's semi-major axis in meters
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f ** 2  # Square of eccentricity

    def geodetic_to_ecef(lat, lon, alt):
        """Convert geodetic coordinates (latitude, longitude, altitude) to ECEF."""
        lat = np.radians(lat)
        lon = np.radians(lon)
        
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - e2) + alt) * np.sin(lat)
        
        return X, Y, Z

    # Convert origin geodetic coordinates to ECEF
    origin_xyz = geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    
    # Output array: time, x, y, z
    xyz_data = np.zeros((latlon_data.shape[0], 3))
    
    for i in range(latlon_data.shape[0]):
        
        # Convert geodetic coordinates to ECEF
        xyz_ecef = geodetic_to_ecef(*latlon_data[i])
        
        # Compute relative xyz coordinates
        xyz = np.subtract(xyz_ecef,origin_xyz)
        
        xyz_data[i] = xyz
    
    return xyz_data

def convert_LatLonAlt_to_Grid(GRIDCOORD_LatLonAlt, ORIGIN_LatLonAlt, SHAPE_REAL, SHAPE_GRID):
    COORD_LIST = convert_latlon_to_xyz(GRIDCOORD_LatLonAlt, *ORIGIN_LatLonAlt) # z, y, x [m]
    COORD_LIST = COORD_LIST[:,[2,1,0]]/1000 # x, y, z [km]
    COORD_LIST[:,0] /= 1.45 # Scale the x-axis to fit the grid
    COORD_LIST[:,0] += SHAPE_REAL[0] / 2 # Shift the axes
    COORD_LIST[:,1] += SHAPE_REAL[1] / 2
    COORD_LIST[:,2] += SHAPE_REAL[2] / 2 + 1

    COORD_LIST = [tuple(convertCoord_RealToGrid(coord, SHAPE_REAL, SHAPE_GRID)) for coord in COORD_LIST]

    return COORD_LIST 

def save_txt(data, filename): 
    np.set_printoptions(threshold=np.inf)
    with open(file=filename, mode='w') as file:
        file.write(str(data))
    np.set_printoptions(threshold=1000)