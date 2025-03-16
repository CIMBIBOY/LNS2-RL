import numpy as np
import matplotlib.pyplot as plt


import sys
import matplotlib

from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import  saveData_Numpy, saveData_Pickle
from GridBasedPathPlanning.Plan import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import PathPlanning_PostProcess
from GridBasedPathPlanning.Environment.GraphicsEngine2D import GraphicsEngine2D

FRAMES = 600

def PathPlanning_TJPS_2D():
    np.random.seed(0)
    grid_static, shape = getSliceMap3D()

    grid_static = grid_static[::2,::2,5]
    grid_static = np.expand_dims(grid_static, axis=2)
    shape = grid_static.shape # (64,64,1)

    grid = GridMapEnv(grid_size=shape)
    grid.SetStaticGrid(grid_static)
    grid_seq = grid.GenerateGridSequence(FRAMES, reset_after=False, clip=False)

    print("Start: ", start := grid.GetRandomFreeCell((10,60,0),r=2)) # fig: 18,146 -> 70,40, 15 wait
    print("Goal: ",  goal  := grid.GetRandomFreeCell((62,26,0),r=2))
    
    planner = TJPS(grid_seq, start, goal,
                    obs_threshold=0.01)
    plan = planner.plan()

    
    folder = 'GridBasedPathPlanning/Data/Processing/plot_trajgen/'
    plans = PathPlanning_PostProcess(grid_seq=grid_seq, plans=[plan],
                                     dronesize=0.4,
                                    correct_gradient_radius=6,
                                    spline_smax=100,
                                    obs_threshold_correct = 0.01,
                                    shift_factor_list = [0.5],
                                    max_correction_limit=2,
                                    interpolate_MinimumSnapTrajectory=True)
    plan = plans[0]
    
    

    plt.rcParams['text.usetex'] = True
    cmap = matplotlib.colormaps['magma_r']

    grid = grid_seq[0]

    fig, ax = plt.subplots()

    # Plot obstacles
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x,y] > 0.01:
                ax.add_patch(plt.Rectangle((x+0.05, y+0.05), 0.9, 0.9, color=cmap(grid[x,y])))
    # Set limits and grid
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_xticks(np.arange(0, grid.shape[0], 5))
    ax.set_yticks(np.arange(0, grid.shape[1], 5))
    ax.set_xticks(np.arange(0, grid.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(0, grid.shape[1], 1), minor=True)
    ax.grid(which='both', linewidth=0.5)
    ax.set_xlabel('x [-]')
    ax.set_ylabel('y [-]')
    ax.set_xlim(0, 60)
    ax.set_ylim(15, 55)

    path = plan['path_extracted'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker='.', color='tab:blue', label='Extracted path')
    path = plan['path_corrected'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker='.', color='tab:red', label='Corrected path')
    path = plan['path_interp_BSpline'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color='tab:green', label='Trajectory fitting')
    path = plan['path_interp_MinimumSnapTrajectory'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color='tab:orange', linestyle = 'dashed', label='Minimum Snap Trajectory')
    
    plt.legend(loc = 'upper right', framealpha=1)
    #plt.show()
    fig.savefig(Path(__file__).parent/'plot_trajgen.pdf',bbox_inches='tight',transparent=True, pad_inches=0)
    #fig.savefig(Path(__file__).parent/'plot_trajgen.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=600)


if __name__ == '__main__':
    PathPlanning_TJPS_2D()
