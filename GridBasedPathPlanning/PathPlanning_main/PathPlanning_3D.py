import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv import getDemoGridMapEnv
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import PathPlanning_PostProcess
from GridBasedPathPlanning.Plan import *
from GridBasedPathPlanning.GraphicsEngine3D import *

PLAN = True
FRAMES = 400
DRONES = 3

ROOTFOLDER = 'GridBasedPathPlanning/Data/Processing/'
SUBFOLDER = 'demo_3drones' # Choose or create a folder inside Data/Processing
FOLDER = ROOTFOLDER + SUBFOLDER + '/'

def PathPlanning_3D():
    '''
    #grid_static, shape = getSliceMap3D(reduction=4)    # Radarmap 3D slices
    grid_static = generateRandomTerrain_Simplex2D(shape:=(170,55,18), scale=max(shape)/2.5, octaves=2, persistence=1.2)   # Random generated terrain

    grid = GridMapEnv(grid_size=shape) # Create the Grid environment
    grid.SetStaticGrid(grid_static)
    #grid.addObjectsFromYAML(sys.path.append(Path(__file__).parents[1])) + '/Env_config/TJPS_3D_base.yaml')
    grid.addRandomDynamicObstacles(no_of_obs=10, d_range=(1,6), alpha = 0.5)

    for peak in getListOfHighestMountains(grid_static, n=5):
        grid.CreateObstacle(typ='dynamic', shape='Radar', data={'p':peak, 'a':11, 'n':1.5})
    '''
    grid = getDemoGridMapEnv()

    plans = []
    for drone_idx in range(DRONES):

        grid_seq= grid.GenerateGridSequence(FRAMES, reset_after=True)
        shape = grid_seq[0].shape

        if drone_idx == 0:  # Save the grid_seq without any planned drones
            saveData_Numpy(FOLDER, grid_seq=grid_seq, grid_static=grid.grid_static)

        
        if PLAN:
            #print("Start: ", start := grid.GetRandomFreeCell((2,46,17), r=5))
            #print("Goal: ",  goal  := grid.GetRandomFreeCell((168,40,16), r=5))
            print("Start: ", start := grid.GetRandomFreeCell((32,2,8), r=5))
            print("Goal: ",  goal  := grid.GetRandomFreeCell((32,126,8), r=5))

            planner = TJPS(grid_seq, start=start, goal=goal,
                            obs_threshold=0.01,
                            goal_threshold=5,
                            max_wait=2)

            plan = planner.plan()
            if plan is not None:
                plans.append(plan)
                # Add the already planned route as a dynamic obs
                grid.CreateObstacle(typ='dynamic',shape='Ellipsoid',data={'d':3, 'path':plan['path_extracted']})
            else:
                print('PLANNING FAILED')

            saveData_Pickle(FOLDER, plans = plans) # Save after every plan step if one fails


if __name__ == '__main__':
    PathPlanning_3D()
    PathPlanning_PostProcess(FOLDER, interpolate_MinimumSnapTrajectory=True)