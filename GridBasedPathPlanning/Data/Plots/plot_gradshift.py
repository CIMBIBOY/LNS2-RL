import sys
import matplotlib

from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import  saveData_Numpy, saveData_Pickle
from GridBasedPathPlanning.Plan import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import PathPlanning_PostProcess

FRAMES = 600

def PathPlanning_TJPS_2D():
    np.random.seed(0)
    shape = (40,20)

    grid = GridMapEnv(grid_size=shape)
    grid.CreateObstacle(typ='static',shape='Ellipsoid', data = {'p':(20,-3),'d':35,'alpha':1.0,'gradient':0.5})
    grid.CreateObstacle(typ='static',shape='Ellipsoid', data = {'p':(0,20),'d':26,'alpha':1.0,'gradient':0.5})
    grid_seq = grid.GenerateGridSequence(FRAMES, reset_after=False, clip=False)

    print("Start: ", start := grid.GetRandomFreeCell((0,8),r=1)) # fig: 18,146 -> 70,40, 15 wait
    print("Goal: ",  goal  := grid.GetRandomFreeCell((39,3),r=1))
    
    planner = TJPS(grid_seq, start, goal,
                    obs_threshold=0.35)
    plan = planner.plan()

    folder = 'GridBasedPathPlanning/Data/Processing/plot_gradshift/'
    saveData_Numpy(folder, grid_seq=grid_seq)
    saveData_Pickle(folder, plans=[plan] )
    PathPlanning_PostProcess(folder,
                             correct_gradient_radius=5,
                             spline_smax=10,
                             obs_threshold_correct = 0.4,
                             shift_factor_list = [2.4],
                             max_correction_limit=6)
    plan = loadData_Pickle(folder)[0]

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
    ax.set_xticks(np.arange(0, grid.shape[0]+1, 10))
    ax.set_yticks(np.arange(0, grid.shape[1]+1, 10))
    ax.set_xticks(np.arange(0, grid.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(0, grid.shape[1], 1), minor=True)
    ax.grid(which='both', linewidth=0.5)
    ax.set_xlabel('x [-]')
    ax.set_ylabel('y [-]')

    path = plan['path_extracted'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker='.', color='tab:blue', label='Extracted path')
    path = plan['path_corrected'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker='.', color='tab:red', label='Corrected path')
    path = plan['path_interp_BSpline'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, color='tab:green', label='Trajectory fitting')

    plt.legend(loc = 'upper right', framealpha=1)
    #plt.show()
    fig.savefig(Path(__file__).parent/'plot_gradshift.pdf',bbox_inches='tight',transparent=True, pad_inches=0)
    #fig.savefig(Path(__file__).parent/'plot_gradshift.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=600)


if __name__ == '__main__':
    PathPlanning_TJPS_2D()
