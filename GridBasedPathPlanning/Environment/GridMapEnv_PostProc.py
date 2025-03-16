import numpy as np
import itertools
from pathlib import Path
from scipy.interpolate import splprep, splev

from GridBasedPathPlanning.MinimumSnapTrajectory import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import loadData_Pickle, saveData_Pickle


def PathPlanning_PostProcess(folder=None, grid_seq=None, plans=None, **kwargs):

    '''
    If you give a folder parameter, then the function loads the data from disk
    If the grid_seq and plans parameter is given, then it does not load from disk.

    ### Parameters:
    - dronesize
    - interpolate_MinimumSnapTrajectory
    - obs_threshold_spline
    - obs_threshold_correct

    - shift_factor_list
    - correct_gradient_radius
    - max_correction_limit

    - spline_smax
    '''

    if folder is not None:
        # Load the grid_seq and plans from disk (acquired during the path planning process)
        grid_seq = np.load(Path(folder)/'grid_seq.npz')['grid_seq']
        plans = loadData_Pickle(folder=folder, file='plans.pkl')
    
    grid_seq[np.isinf(grid_seq)] = 1 # CONVERT INF TO 1

    for plan in plans:
        plan = CorrectAndInterpolatePlan(plan, grid_seq, **kwargs)
        if folder is not None:
            saveData_Pickle(folder=folder, plans=plans)

    return plans  # ADD ALREADY PLANNED AGENTS TO GRID SEQ !!!!!!!!!!
    
    

def CorrectAndInterpolatePlan(plan, grid_seq, **kwargs):
    
    # DEFAULT VALUES
    obs_threshold_correct = kwargs.setdefault('obs_threshold_correct', 0.01)
    obs_threshold_spline  = kwargs.setdefault('obs_threshold_spline', obs_threshold_correct)

    dronesize = kwargs.setdefault('dronesize', 0.3)
    interpolate_MinimumSnapTrajectory = kwargs.setdefault('interpolate_MinimumSnapTrajectory', False)
    shift_factor_list = kwargs.setdefault('shift_factor_list', [2.0, 1.2, 0.8, 0.4, 0.1, 0]) # list[float]
    correct_gradient_radius = kwargs.setdefault('correct_gradient_radius', 3)
    max_correction_limit = kwargs.setdefault('correct_gradient_radius', 10)
    spline_smax = kwargs.setdefault('spline_smax', 1000)


    path_correction_data = PathCorr_calcNegativeGradient(grid_seq, plan['path_extracted'], **kwargs)

    if shift_factor_list[-1] != 0: shift_factor_list.append(0)# shift_factor list must include 0 as last element
    for shift_factor in shift_factor_list:

        if shift_factor == 0: path = plan['path_extracted']
        else:
            plan['path_corrected'] = PathCorr_evalNegativeGradient(grid_seq,
                                                                   plan['path_extracted'],
                                                                   *path_correction_data,
                                                                    shift_factor, **kwargs)
            path = plan['path_corrected']

        collisionNum_Spline_s = lambda s: collisionNum_Spline(s,path,grid_seq,dronesize,obs_threshold=obs_threshold_spline)
        collisionNum_woInterp = collisionNum_Spline_s(0)

        print(f'Shift factor: {shift_factor} \tCollision number without interpolation: {collisionNum_woInterp}')

        if collisionNum_woInterp == 0:
            ''' The function we want to find roots for has a value set of : [0, inf) and is stricly increasing
                So we have to shift it 'down' with a shift parameter '''
            
            shift = -0.1
            try:
                s_optimal = bisection_method(shift_decorator(collisionNum_Spline_s,shift=shift), 0, spline_smax, tol = 0.5)
            except ValueError as e:
                print(e)

                # If the function is positive at s_min -> Collision with current shift_factor
                if e.args[1]['f_a'] > shift:
                    continue # Continue with the next shift_param

                # If the function is zero at s_max -> Even s_max does not result in collision
                elif e.args[1]['f_b'] == shift: 
                    s_optimal = spline_smax # Use the maximum smoothness parameter
            finally:
                print(f'Optimal s param: {s_optimal:.1f} Collision num: {collisionNum_Spline_s(s_optimal)}')

                interpdata_tcku = splprep(path.T, s=s_optimal)
                plan['path_interp_BSpline_tcku'] = interpdata_tcku
                plan['path_interp_BSpline'] = evalBSpline(*interpdata_tcku, no_points=path.shape[0]*5, calc_rot=True)
                break
        else:
            continue

    if interpolate_MinimumSnapTrajectory:
        path_interpolated_BSpline = evalBSpline(*interpdata_tcku, no_points=path.shape[0]) # Same number of points as path_extracted
        plan['path_interp_MinimumSnapTrajectory'] = MinimumSnapTrajectory(path = path_interpolated_BSpline[:,1:4]).T

    return plan



def PathCorr_calcNegativeGradient(grid_seq:np.ndarray, path_extracted:np.ndarray, **kwargs):
    '''This function calculates the negative gradients, and determines their maximum value which does not result in collision'''

    radius = kwargs.setdefault('correct_gradient_radius', 1) # int
    max_correction_limit = kwargs.setdefault('max_correction_limit', 2)

    path_correction_neggrad = np.zeros((path_extracted.shape[0],path_extracted.shape[1]-1))
    path_correction_limit = np.zeros((path_extracted.shape[0]))

    # examine a R sized area around every point of the path_extracted
    for time in range(len(path_extracted)):
        point = path_extracted[time,1:]
        bound_min = np.clip(point - np.full_like(point,radius), np.zeros_like(point), np.array(grid_seq.shape[1:])-1).astype(int) # limit the examined area to the array size
        bound_max = np.clip(point + np.full_like(point,radius), np.zeros_like(point), np.array(grid_seq.shape[1:])-1).astype(int)
        bound_box_ranges = [range(bound_min[i],bound_max[i]+1) for i in range(len(point))]
        bound_box_cellnum = np.prod(bound_max - bound_min + 1)
        coordinates = itertools.product(*bound_box_ranges)
        grad = np.zeros(len(point))
        grad_sum = 0
        for coord in coordinates:
            direction = np.subtract(coord,point)
            if any(direction): # Not the center cell
                grad = grad + ((direction / np.linalg.norm(direction)**2) * grid_seq[time, *coord]) # normalize(direction) * weight
                grad_sum += grid_seq[time, *coord]

        grad_neg = - grad  * (grad_sum / bound_box_cellnum)
        # Ahol kitöltetlen a tér, de kis értékek vannak, oda nagy eltolás kéne
        # ahol nagyon kitöltött a tér, mindegy milyen értékeknél ott kis eltolások kellenek hogy ne ütközzünk

        collision_limit = detectCollision_inDirection(grid_seq=grid_seq, point_origin=path_extracted[time], 
                                                      direction= grad_neg, distance_limit=max_correction_limit, **kwargs)
        
        path_correction_neggrad[time] = grad_neg
        path_correction_limit[time] = collision_limit

    return path_correction_neggrad, path_correction_limit

def PathCorr_evalNegativeGradient(grid_seq,
                                    path_extracted:np.ndarray, 
                                    path_correction_neggrad:np.ndarray, 
                                    path_correction_limit:np.ndarray, 
                                    shift_factor=0, **kwargs):
    '''This function evaluates the corrected path based on the already calculated gradients and the shift factor'''

    path_corrected = np.empty_like(path_extracted)
    for time in range(len(path_extracted)):

        grad_neg = path_correction_neggrad[time]
        collision_limit = path_correction_limit[time]

        point_corrected = path_extracted[time,1:] + grad_neg * shift_factor
        if np.linalg.norm(point_corrected - path_extracted[time,1:]) > collision_limit:
            grad_unit = grad_neg/np.linalg.norm(grad_neg) if any(grad_neg) else np.zeros_like(grad_neg)
            point_corrected = path_extracted[time,1:] + grad_unit * collision_limit

        point_corrected = np.clip(point_corrected, 0, grid_seq.shape[1:]) # clip into grid bounds
        path_corrected[time] =  [time, *point_corrected]

    return path_corrected

def detectCollision_inDirection(grid_seq:np.ndarray, point_origin, direction, distance_limit, **kwargs):
    '''The function determines if a drone with a specified size collides with an object or not, 
       when traversing from point_origin(t,x,y,z) in the direction D, with a max distance specified.
       It returns the last point which did not result in a collision.
       point_origin: (t,x,y,z)'''

    dronesize = kwargs.setdefault('dronesize', 0.5)
    obs_threshold = kwargs.setdefault('obs_threshold_correct', 0.01)

    # Go from point in the specified direction in small linear steps and check for collision, stop when one occurs
    step_len = dronesize/3
    step_num = int(np.ceil(distance_limit/step_len))
    direction_unit = direction/np.linalg.norm(direction) if any(direction) else np.zeros_like(direction)
    
    for step in range(step_num):
        point_check = point_origin[1:] + (step+1) * direction_unit * step_len # 0 offset is always correct, so no need to check
        collision = detectCollision(grid_seq[int(point_origin[0])], point_check, dronesize, obs_threshold)

        if collision:
            collision_limit = step * step_len # If the current shift is too much, go back one step
            return  collision_limit

    # If there is no collision even at the last step, the collision limit is calculated from the distance_limit
    collision_limit = distance_limit
    return collision_limit


def detectCollision(grid:np.ndarray, point:np.ndarray, dronesize: float, obs_threshold:float=0.01, return_collision_max_value = False) -> bool:
    '''The function determines if a drone with a specified size collides with an object or not,
       THE INPUT GRID'S TIME MUST CORRESPOND WITH THE DRONE'S POSITION WHEN USING THIS FUNCTION,
       since this function is not able to determine if the given points time by default'''
    point_min = np.clip(np.floor(point + 0.5 - dronesize/2), a_min=0, a_max=np.array(grid.shape)-1).astype(int)
    point_max = np.clip(np.floor(point + 0.5 + dronesize/2), a_min=0, a_max=np.array(grid.shape)-1).astype(int)
    drone_slice = tuple(slice(point_min[i],point_max[i]+1) for i in range(len(point_min)))
    # print(point, point_min, point_max, drone_slice)
    if not return_collision_max_value:
        return np.any(grid[drone_slice] >= obs_threshold) # Return only the number of collisions ( >= obs_threshold)
    else:
        return np.any(grid[drone_slice] >= obs_threshold), np.max(grid[drone_slice])
    
def collisionNum(grid_seq:np.ndarray, path:np.ndarray, dronesize:float, obs_threshold:float=0.01, print_collision_warning = True) -> int:
    collision_num_above_obs_threshold = 0
    for point in path:
        t_round = round(point[0])
        point_coord = point[1:len(point)]

        collision, collision_max_value = detectCollision(grid_seq[t_round], point_coord, dronesize, obs_threshold, return_collision_max_value=True)

        if collision:
            collision_num_above_obs_threshold += 1
            
            if print_collision_warning: print(f"COLLISION: t: {t_round}, p: {point_coord}, val: {collision_max_value}")
    return collision_num_above_obs_threshold

def evalBSpline(tck, u, no_points = 100, calc_rot = True):
    eval_points = np.linspace(0,1,num=no_points)
    path_interp_pos = np.array(splev(eval_points, tck)).T
    if calc_rot:
        path_interp_tangent = np.array(splev(eval_points, tck, der=1)).T[:,1:4]
        path_interp_rot = convertTangentToEuler(path_interp_tangent)
        path_interp = np.hstack((path_interp_pos, path_interp_rot))
    else:
        path_interp = path_interp_pos

    return path_interp

def convertTangentToEuler(tangents): # rows: tangent vector (x,y,z)
    euler_angles = np.empty(shape=tangents.shape)
    up_global = np.array([0, 0, 1])

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
    
    def euler_from_rotation_matrix(R):
        """Convert a rotation matrix into Euler angles (roll, pitch, yaw)."""
        if R[2, 0] < 1:
            if R[2, 0] > -1:
                yaw = np.arctan2(R[1, 0], R[0, 0])
                pitch = np.arcsin(-R[2, 0])
                roll = np.arctan2(R[2, 1], R[2, 2])
            else:  # R[2, 0] = -1
                yaw = np.arctan2(-R[1, 2], R[1, 1])
                pitch = -np.pi / 2
                roll = 0
        else:  # R[2, 0] = +1
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.pi / 2
            roll = 0
            
        return roll, pitch, yaw

    r = np.zeros(shape=(3,3))

    for i in range(tangents.shape[0]):
        tangent = tangents[i]
        tangent  = normalize(tangent)
        right    = normalize(np.cross(up_global, tangent))
        up_local = normalize(np.cross(tangent, right))

        # Rotation matrix # COLUMNS: right, up_local, tangent
        r[:,0] = right
        r[:,1] = up_local
        r[:,2] = tangent

        euler_angles[i] = euler_from_rotation_matrix(r)    # Convert to Euler angles (roll, pitch, yaw) in radians

    return euler_angles

def collisionNum_Spline(s, path, grid_seq, dronesize, obs_threshold = 0.01):
    interpdata_tcku = splprep(path.T, s=s)
    path_interpolated = evalBSpline(*interpdata_tcku, no_points=path.shape[0], calc_rot=False)
    path_collision_num_above_th = collisionNum(grid_seq, path_interpolated, dronesize, obs_threshold, print_collision_warning=False)
    return path_collision_num_above_th

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Finds the root of a function using the bisection method.

    Parameters:
        func (function): The function for which the root is to be found.
        a (float): The left endpoint of the initial interval.
        b (float): The right endpoint of the initial interval.
        tol (float): The tolerance for the root.
        max_iter (int): The maximum number of iterations.

    Returns:
        float: The approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        args = {'f_a': f(a), 'f_b': f(b)}
        raise ValueError("The function does not change sign over the interval", args)

    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        midpoint = (a + b) / 2
        if f(midpoint) == 0:
            return midpoint
        elif f(midpoint) * f(a) < 0:
            b = midpoint
        else:
            a = midpoint
        iter_count += 1

    return (a + b) / 2

def shift_decorator(func, shift):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # Call the original function with its arguments
        return result + shift           # Shift the result by the shift parameter
    return wrapper

def MinimumSnapTrajectory(path, no_points_mult = 2): # no_points_mult default: FPS/FPS_animation
    # Interpolate path Minimum Snap Trajectory
    # num: number of points to interpolate to
    poly_deg = 3
    optim_order = 3
    time_ori, coeff, poly = getMinimumSnapTrajectory(path, poly_deg, optim_order)
    x_f = getStateFromDifferentialFlatness(time_ori, coeff, poly_deg)

    time_interp = np.linspace(0, time_ori[-1], int((path.shape[0])*no_points_mult)) # set interpolation timestep
    x = np.array(x_f(np.expand_dims(time_interp, 0)))           # shape: num_states * t.shape[0]
    x = np.vstack([np.expand_dims(np.array(time_interp), 0),x]) # x: t, x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz
    return x


