from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GraphicsEngine2D import GraphicsEngine2D

from GridBasedPathPlanning.Plan.Dijkstra_mod import Node_Dijkstra, dijkstra, extractPaths_Dijkstra
from GridBasedPathPlanning.Plan.Dijkstra_mod import motions as motions_Dijkstra
from GridBasedPathPlanning.Plan.WJPS_2D import WJPS

SYM_ENV = 'mpl'
PLAN = 'WJPS'
FRAMES = 1
FPS_ANIMATION = 1

def InitGrid(grid):
    
    # Create STATIC obstacles
    #grid.CreateObstacle(typ='static',shape='Ellipsoid', data = {'p':(3,4),'d':11,'alpha':0.8,'gradient':0.9})
    grid.CreateObstacle(typ='static',shape='rect', data = {'p_min':(0,1),'p_max':(5,5),'alpha':9})

    grid.RenderAllObstacles(typ='static')
    #grid.RenderAllObstacles(typ='dynamic')
    grid.RenderGrid(clip = False)
    return grid


def PathPlanning_WJPS_2D():
    
    shape = (10,10)    # EMPTY MAP WITH OBS
    print("Start: ", start := (0,0))
    print("Goal: ",  goal := (7,7))

    grid = GridMapEnv(grid_size=shape) # Create the Grid environment
    grid.CreateObstacle(typ='static',shape='rect',data = {'p_min':(0,0), 'p_max':shape, 'alpha':1})
    #grid.SetStaticGrid(grid_static=grid_static)
    grid = InitGrid(grid)

    print(grid.grid)

    path_dijkstra_all_nodes = dijkstra(grid.grid,Node_Dijkstra(start),motions_Dijkstra)
    path_dijkstra = extractPaths_Dijkstra(path_dijkstra_all_nodes,motions_Dijkstra)
    for i in path_dijkstra[goal]: print(i)

    planner = WJPS(start=start, goal=goal, grid=grid)
    path, path_extracted, cost = planner.plan() if PLAN == 'WJPS' else None

    # PLOT
    if SYM_ENV == 'mpl':
        pass
        plotwMatplotLib_2D(np.expand_dims(grid.grid, axis=0), path_extracted, FRAMES, PLAN)
    elif SYM_ENV == 'pg':
        app = GraphicsEngine2D(np.expand_dims(grid.grid, axis=0), path_extracted, FPS_ANIMATION)
        app.run()
