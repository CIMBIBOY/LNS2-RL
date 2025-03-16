import sys
from pathlib import Path
import csv
import datetime
import json

sys.path.append(str(Path(__file__).parents[2]))
from GridBasedPathPlanning.Environment.GridMapEnv import *
from GridBasedPathPlanning.Environment.GridMapEnv_Utils import getBenchmarkMap
from GridBasedPathPlanning.Plan import *
from GridBasedPathPlanning.Environment.GridMapEnv_PostProc import collisionNum

np.set_printoptions(legacy='1.13')

# C:/Users/adamk/Python/SZTAKI/Scripts/python.exe c:/Users/adamk/OneDrive/SZTAKI/CODE/GridBasedPathPlanning/PathPlanning_main/PathPlanning_Benchmark.py

def PathPlanning_Benchmark():

    np.random.seed(0)

    BENCHMAP = 0
    MULTISTEP = True
    PLOT = False
    
    if not BENCHMAP: # EMPTY MAP
        shape_0 = (200,200)
    else:
        benchmap_name = "Paris_1_256.map" # Berlin_0_256.map"
        grid_static, shape_0 = getBenchmarkMap(benchmap_name, reduction=1)
        #shape_0 = (shape_0[1], shape_0[0]) # only for warehouse

    iter_env = 3          # Number of different random envs to test
    iter_start_goal = 10   # Number of different start goal pairs to test in the same env

    # Define a list to determine how many obstacles to add in each new run
    obj_add = [0]
    obj_add.extend([2]*5)
    obj_add.extend([5]*8)
    obj_add.extend([10]*15) # 200 obs, 30 runs

    total_iter_num = iter_env * iter_start_goal * len(obj_add)


    plan_info = {}
    plan_info['shape'] = shape_0
    if BENCHMAP: plan_info['benchmap'] = benchmap_name
    plan_info['comment'] = f'multistep planning'
    plan_info['obstacle_d_range'] = (d_range := (1,10))
    plan_info['TJPS_max_wait'] = (TJPS_max_wait := 1)
    plan_info['max_execution_time'] = (max_execution_time := 30)
    plan_info['corridor_diameter'] = 30

    # FILE MANAGEMENT:
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    filepath = 'GridBasedPathPlanning/Data/Benchmark/'
    
    with open(filepath+filename,mode='w') as file:
        file_writer = csv.writer(file, delimiter='\t',lineterminator='\n')
        plan_info_str = json.dumps(plan_info)
        file.write(plan_info_str + '\n\n')
        file_writer.writerow(['iter','plan','step','shape','start','goal','no_of_obs','time_to_plan','cost','start_goal_dist'])
        file.flush()

        iter_num = 0
        for env in range(iter_env):

            grid = GridMapEnv(grid_size=shape_0)
            if BENCHMAP: grid.SetStaticGrid(grid_static, convert_nonzero_to_inf=True)

            for obj_num in obj_add:
            
                if obj_num!=0: grid.addRandomDynamicObstacles(no_of_obs=obj_num, d_range=d_range, alpha=np.inf)
                grid_seq_0 = grid.GenerateGridSequence(FRAMES=1000, reset_after=True)
                if MULTISTEP:
                    slices = tuple(slice(None, None, 2) for _ in range(grid_seq_0.ndim))
                    grid_seq_1 = grid_seq_0[slices]
                    shape_1 = grid_seq_1.shape[1:]

                for start_goal in range(iter_start_goal):
                    iter_num += 1

                    start_0 = grid.GetRandomFreeCell(force_even=MULTISTEP) # Only force even coord if multistep is enabled
                    goal_0 = grid.GetRandomFreeCell(force_even=MULTISTEP)
                    
                    if MULTISTEP:
                        start_1 = tuple(int(x/2) for x in start_0)
                        goal_1  = tuple(int(x/2) for x in goal_0)

                    start_goal_dist = np.linalg.norm(np.subtract(goal_0,start_0))
                    print(f'Run {iter_num}/{total_iter_num}\tStart {tuple(map(int,start_0))}\tGoal {tuple(map(int,goal_0))}\tDist {start_goal_dist:.1f}')                        


                    print("STEP FULL:", shape_0, start_0, goal_0)
                    planner = TJPS(grid_seq_0, start=start_0, goal=goal_0, goal_threshold=3,
                                    max_wait=TJPS_max_wait, log = 0, max_execution_time=max_execution_time)
                    try:
                        plan = planner.plan()
                    except IndexError:
                        plan = None

                    if plan is None:
                        print('PATH NOT FOUND BY TJPS')
                    else:
                        collision = collisionNum(grid_seq_0, plan['path_extracted'],dronesize=0.1)
                        if collision: print(f'Number of collisions with TJPS: {collision}')

                        if MULTISTEP: 
                            print("STEP REDUCED:", shape_1, start_1, goal_1)
                            planner = TJPS(grid_seq_1, start=start_1, goal=goal_1, goal_threshold=1,
                                            max_wait=TJPS_max_wait, log = 0, max_execution_time=max_execution_time)
                            try: 
                                plan_1 = planner.plan()
                            except IndexError:
                                plan_1 = None

                            if plan_1 is not None: 
                                print("STEP BOUND:", shape_0, start_0, goal_0)
                                path_extracted_0_init = plan_1['path_extracted'] * 2    # Upscale the resulting path
                                grid_seq_0_bounded = GenerateBoundedGridSequence(shape_0, path_extracted_0_init, grid_seq_0, corridor_diameter = plan_info['corridor_diameter'])

                                planner = TJPS(grid_seq_0_bounded, start=start_0, goal=goal_0, goal_threshold=3,
                                                max_wait=TJPS_max_wait, log = 0, max_execution_time=max_execution_time)
                                try:
                                    plan_0 = planner.plan()
                                except IndexError:
                                    plan_0 = None

                    if plan is not None and (not MULTISTEP or (plan_1 is not None and plan_0 is not None)):
                        file_writer.writerow([iter_num,'TJPS','FULL',shape_0,start_0,goal_0,len(grid.objects),plan['time_to_plan'],plan['cost'],start_goal_dist])
                        file_writer.writerow([iter_num,'TJPS','REDUCED',shape_1,start_1,goal_1,len(grid.objects),plan_1['time_to_plan'],plan_1['cost'],start_goal_dist/2])
                        file_writer.writerow([iter_num,'TJPS','BOUND',shape_0,start_0,goal_0,len(grid.objects),plan_0['time_to_plan'],plan_0['cost'],start_goal_dist])
                        file_writer.writerow([iter_num,'TJPS','COMBINED',shape_0,start_0,goal_0,len(grid.objects),plan_0['time_to_plan']+plan_1['time_to_plan'],plan_0['cost'],start_goal_dist])
                        file.flush()


                        planner = TAStar(grid_seq_0, start=start_0, goal=goal_0, goal_threshold=0,
                                        diag=False, log = 0, max_execution_time=max_execution_time)
                        try:
                            plan_TAStar = planner.plan()
                        except IndexError:
                            plan_TAStar = None

                        if plan_TAStar is None: print('PATH NOT FOUND BY TAStar')
                        else:
                            file_writer.writerow([iter_num,'TAStar','FULL',shape_0,start_0,goal_0,len(grid.objects),plan_TAStar['time_to_plan'],plan_TAStar['cost'],start_goal_dist])
                            file.flush()


if __name__ == '__main__':
    PathPlanning_Benchmark()