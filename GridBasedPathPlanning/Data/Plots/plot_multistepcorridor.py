import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[3]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import getBenchmarkMap
from GridBasedPathPlanning.Environment.GraphicsEngine2D import GraphicsEngine2D
from GridBasedPathPlanning.Plan import *

PLAN = True
FRAMES = 1000

# BEFORE USE: Uncomment the return line in the GenerateBoundedGridSequence function

def PathPlanning_TJPS_2D():
    np.random.seed(0)
    grid_static, shape_0 = getBenchmarkMap("Paris_1_256.map", reduction=2)

    grid = GridMapEnv(grid_size=shape_0) # Create the Grid environment
    grid.SetStaticGrid(grid_static=grid_static, convert_nonzero_to_inf=True)
    #grid.addRandomDynamicObstacles(no_of_obs=10, d_range=(1,20), alpha = 0.3)

    grid_seq_0 = grid.GenerateGridSequence(FRAMES, reset_after=False, clip=False)
    slices = tuple(slice(None, None, 2) for _ in range(grid_seq_0.ndim))
    grid_seq_1 = grid_seq_0[slices]
    shape_1 = grid_seq_1.shape[1:]

    start_0 = grid.GetRandomFreeCell((2,0),r=5, force_even=True)
    goal_0  = grid.GetRandomFreeCell((126,120), r=5, force_even=True)
    start_1 = tuple(int(x/2) for x in start_0)
    goal_1  = tuple(int(x/2) for x in goal_0)
    
    print("STEP REDUCED:", shape_1, start_1, goal_1)
    planner = TJPS(grid_seq_1, start=start_1, goal=goal_1,
                   goal_threshold=1,
                   max_wait=0)
    plan_1 = planner.plan()

    print("STEP BOUND:", shape_0, start_0, goal_0)
    path_extracted_0_init = plan_1['path_extracted'] * 2    # Upscale the resulting path
    grid_seq_0_bounded, grid_corridor = GenerateBoundedGridSequence(shape_0, path_extracted_0_init, grid_seq_0, 
                                                                    corridor_diameter = 50,
                                                                    corridor_diameter_div=2,
                                                                    return_grid_corridor=True)

    planner = TJPS(grid_seq_0_bounded, start=start_0, goal=goal_0, 
                   goal_threshold=1)
    plan_0 = planner.plan()




    # PLOT
    plt.rcParams['text.usetex'] = True

    grid = grid_seq_0[0]
    grid_corridor = grid_corridor

    fig, ax = plt.subplots()

    # Plot obstacles
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] >= 0.1:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

    # Plot corridor
    for x in range(grid_corridor.shape[0]):
        for y in range(grid_corridor.shape[1]):
            if grid_corridor[x, y] >= 0.1:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='tab:blue', alpha=0.25, edgecolor=None))

    # Set limits and grid
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_xticks(np.arange(0, grid.shape[0]+1, 16))
    ax.set_yticks(np.arange(0, grid.shape[1]+1, 16))
    #ax.grid(which='both')
    ax.set_xlabel('x [-]')
    ax.set_ylabel('y [-]')

    
    path = path_extracted_0_init[:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker='.', color='tab:blue', label='TJPS Reduced')
    path = plan_0['path_extracted'][:,1:3]
    ax.plot(path[:, 0] + 0.5, path[:, 1] + 0.5, marker=' ', color='tab:red', label='TJPS Bounded')
    
    
    handles, labels = ax.get_legend_handles_labels()    # access legend objects automatically created from data
    rect = plt.Rectangle((x, y), 1, 1, facecolor='tab:blue', alpha=0.25, edgecolor=None, label='Corridor')
    handles.append(rect)  # add manual symbols to auto legend

    plt.legend(loc = 'lower right', framealpha=1, handles=handles)
    #plt.show()
    fig.savefig(Path(__file__).parent/'plot_corridor.pdf',bbox_inches='tight',transparent=True, pad_inches=0) 
    #fig.savefig(Path(__file__).parent/'plot_corridor.png',bbox_inches='tight',transparent=True, pad_inches=0, dpi=600)   



if __name__ == '__main__':
    PathPlanning_TJPS_2D()