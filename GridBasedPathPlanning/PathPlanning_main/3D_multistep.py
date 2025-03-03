import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.GraphicsEngine3D import *
from GridBasedPathPlanning.Plan import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import PathPlanning_PostProcess

np.set_printoptions(legacy='1.13')

PLAN = True
FRAMES = 400
DRONES = 3

ROOTFOLDER = 'GridBasedPathPlanning/Data/Processing/'
SUBFOLDER = 'multistep' # Choose or create a folder inside Data/Processing
FOLDER = ROOTFOLDER + SUBFOLDER + '/'

START = (4,76,28)
GOAL = (216,55,24)
START_RADIUS = (10,10,5)
GOAL_RADIUS = (10,10,5)


def PathPlanning_3D():

    np.random.seed(0)

    grid_static = generateRandomTerrain_Simplex2D(shape_0:=(220,80,40), scale=max(shape_0)/2.0, octaves=2, persistence=1.2)   # Random generated terrain

    grid = GridMapEnv(grid_size=shape_0) # Create the Grid environment
    grid.SetStaticGrid(grid_static, convert_nonzero_to_inf=True)
    grid.addRandomDynamicObstacles(no_of_obs=20, d_range=(1,6), alpha = 0.5)
    for peak in getListOfHighestMountains(grid_static, n=5):
        grid.CreateObstacle(typ='dynamic', shape='Radar', data={'p':peak, 'a':20, 'n':1.5}) # 19

    plans = []
    for drone_idx in range(DRONES):

        grid_seq_0 = grid.GenerateGridSequence(FRAMES)
        slices = tuple(slice(None, None, 2) for _ in range(grid_seq_0.ndim))
        grid_seq_1 = grid_seq_0[slices]
        if drone_idx == 0:
            saveData_Numpy(folder=FOLDER, grid_seq=grid_seq_0, grid_static=grid.grid_static)

        start_0 = grid.GetRandomFreeCell(START, START_RADIUS, force_even=True)
        goal_0 = grid.GetRandomFreeCell(GOAL,   GOAL_RADIUS,  force_even=True)
        start_1 = tuple(int(x/2) for x in start_0)
        goal_1  = tuple(int(x/2) for x in goal_0)


        if PLAN:

            print("STEP 1:", grid_seq_1.shape[1:], start_1, goal_1)
            planner = TJPS(grid_seq_1, start=start_1, goal=goal_1,
                            obs_threshold=0.1, goal_threshold=1,max_wait=3)
            plan_1 = planner.plan()

            if plan_1 is not None:
                print("STEP 0:", shape_0, start_0, goal_0)
                path_extracted_0_init = plan_1['path_extracted'] * 2    # Upscale the resulting path
                grid_seq_0 = GenerateBoundedGridSequence(shape_0, path_extracted_0_init, grid_seq_0, corridor_diameter = 20)

                planner = TJPS(grid_seq_0, start=start_0, goal=goal_0,
                                obs_threshold=0.1, goal_threshold=3, max_wait=3)
                plan_0 = planner.plan()
                
                if plan_0 is not None:
                    plans.append(plan_0)
                    # Add the already planned route as a dynamic obs
                    grid.CreateObstacle(typ='dynamic',shape='Ellipsoid',data={'d':3, 'path':plan_0['path_extracted']}) 
                    saveData_Pickle(FOLDER, plans=plans)


    if PLAN:
        PathPlanning_PostProcess(FOLDER, interpolate_MinimumSnapTrajectory=True)


if __name__ == '__main__':
    #PathPlanning_3D()
    PathPlanning_PostProcess(FOLDER, interpolate_MinimumSnapTrajectory=True, correct_gradient_radius=8, obs_threshold_spline=0.26)
    