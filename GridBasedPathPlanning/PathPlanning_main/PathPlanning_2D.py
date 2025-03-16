import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import getBenchmarkMap
from GridBasedPathPlanning.Environment.GraphicsEngine2D import GraphicsEngine2D
from GridBasedPathPlanning.Plan import *

PLAN = True
FRAMES = 600

def PathPlanning_TJPS_2D():
    np.random.seed(0)
    grid_static, shape = getBenchmarkMap("lak303d.map")

    grid = GridMapEnv(grid_size=shape) # Create the Grid environment
    grid.SetStaticGrid(grid_static=grid_static, convert_nonzero_to_inf=True)
    grid.addRandomDynamicObstacles(no_of_obs=100, d_range=(6,14), alpha = 0.3)
    grid_seq = grid.GenerateGridSequence(FRAMES, reset_after=False, clip=False)

    print("Start: ", start := grid.GetRandomFreeCell((18,146),r=10)) # fig: 18,146 -> 70,40, 15 wait
    print("Goal: ",  goal  := grid.GetRandomFreeCell((70,40),r=10))
    
    planner = TJPS(grid_seq, start, goal,
                    obs_threshold=0.01,
                    goal_threshold=3,
                    max_wait=0,
                    max_execution_time=200)
    
    #planner = TAStar(grid_seq, start, goal, goal_threshold=3)
    
    plan = planner.plan() if PLAN else None


    # PLOT
    app = GraphicsEngine2D(grid_seq, plan['path_extracted'], win_size=(1000,1000))
    app.run()

if __name__ == '__main__':
    PathPlanning_TJPS_2D()