import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parents[3]))
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import saveData_Pickle, loadData_Pickle
from GridBasedPathPlanning.Environment.GridMapEnv_Utils_DEMO import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import PathPlanning_PostProcess
from GridBasedPathPlanning.Plan.TJPS import TJPS
from GridBasedPathPlanning.GraphicsEngine3D import *

DRONES = 5 # NUMBER OF DRONES
FOLDER = Path('GridBasedPathPlanning/Data/Processing/demo_5drones')
SHAPE_REAL = (50,100,6)  # [km]
SHAPE_GRID = (64,128,16) # [grid coord system unit]
# DO NOT CHANGE ANYTHING BEFORE THIS LINE

# PARAMETERS -> These values can be changed
ORIGIN_LatLonAlt = (0,0,0)

RANDOM_START_GOAL = False
if not RANDOM_START_GOAL:
    # If you get an error that states that the Start and Goal cells are in an obstacle area please
    # modify the coordinates or give a greater r value.

    STARTGOAL_LatLonAlt = np.loadtxt(FOLDER/'input_data.txt') # Read the start and goal coordinates from the file
    START_LIST = convert_LatLonAlt_to_Grid(STARTGOAL_LatLonAlt[:,:3], ORIGIN_LatLonAlt, SHAPE_REAL, SHAPE_GRID)
    GOAL_LIST = convert_LatLonAlt_to_Grid(STARTGOAL_LatLonAlt[:,3:], ORIGIN_LatLonAlt, SHAPE_REAL, SHAPE_GRID)
    START_RADIUS = 3
    GOAL_RADIUS = 3
else: 
    START = (32,2,8)
    GOAL = (32,126,8)
    START_RADIUS = (32,6,6)
    GOAL_RADIUS = (32,6,6)

TIMESTEP_OF_WAYPOINT_INTERPOLATION = 5  # [s]
OPEN_GRAPHICS_ENVIRONMENT = True


def PathPlanning_DEMO():
    grid_seq = np.load(FOLDER/'grid_seq.npz')['grid_seq']
    plans = []
    for drone_idx in range(DRONES):

        if RANDOM_START_GOAL:
            print("Start: ", start := getRandomFreeCell_DEMO(grid_seq, START, r=START_RADIUS, max_attempts=50))
            print("Goal: ",  goal  := getRandomFreeCell_DEMO(grid_seq, GOAL,  r=GOAL_RADIUS, max_attempts=50))
        else: 
            print("Start: ", start := getRandomFreeCell_DEMO(grid_seq, START_LIST[drone_idx], r=START_RADIUS, max_attempts=50))
            print("Goal: ",  goal  := getRandomFreeCell_DEMO(grid_seq, GOAL_LIST[drone_idx],  r=GOAL_RADIUS, max_attempts=50))

        planner = TJPS(grid_seq,start=start,goal=goal,goal_threshold=5,max_wait=1, )
        
        plan = planner.plan()
        plans.append(plan)
        addObsToGridSeq_DEMO(grid_seq, plan['path_extracted'])
        saveData_Pickle(folder = str(FOLDER), plans = plans)
    
    PathPlanning_PostProcess(FOLDER, grid_seq, plans, correct_gradient_radius=5, obs_threshold_spline=0.26)

    # Convert the waypoints from grid coordinate system to real
    waypoints_xyz_list = []
    waypoints_latlon_list = []
    plans = loadData_Pickle(FOLDER, file='plans.pkl')
    for i in range(DRONES):
        waypoints_xyz = convertWayPointsToSI_DEMO(plans[i], v_knots=120, timestep_real=TIMESTEP_OF_WAYPOINT_INTERPOLATION, shape_real=SHAPE_REAL, shape_grid=SHAPE_GRID)
        waypoints_xyz_list.append(waypoints_xyz)
        waypoints_latlon = convert_xyz_to_latlon(waypoints_xyz,*ORIGIN_LatLonAlt)
        waypoints_latlon_list.append(waypoints_latlon)
    save_txt(waypoints_xyz_list,    FOLDER/'waypoints_txyz.txt')
    save_txt(waypoints_latlon_list, FOLDER/'waypoints_tlatlonalt.txt')

    if OPEN_GRAPHICS_ENVIRONMENT:
        app = GraphicsEngine()
        app.run()

if __name__ == '__main__':
    PathPlanning_DEMO()
    