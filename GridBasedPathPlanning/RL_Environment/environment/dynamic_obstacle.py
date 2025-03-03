import numpy as np
import random
from GridBasedPathPlanning.Plan.TJPS import TJPS
from GridBasedPathPlanning.RL.helpers.utils import save_array_as_png

'''
1.	Initialize dynamic objects on the map.
2.	Move dynamic obstacles based on the A* algorithm or other logic.
3.	Update coordinates of obstacles during simulation steps.
'''

# Function to initialize dynamic obstacles on the map
def initialize_objects(arr, n_dynamic_obst = 20, rng=None):
    """
    Input: array of initial map, number of dynamic obstacles

    Output: array of initial positions of all dynamic obstacles and images after adding dynamic obstacles

    """
    arr = arr.copy()
    coord = []
    h, w = arr.shape[:2]

    if rng is None:
        rng = np.random  # Use the global numpy RNG if none is provided

    while n_dynamic_obst > 0:
        h_obs = rng.integers(0, h)
        w_obs = rng.integers(0, w)

        cell_coord = arr[h_obs, w_obs]
        if cell_coord[0] != 0 and cell_coord[1] != 0 and cell_coord[2] != 0:
            arr[h_obs, w_obs] = [255, 165, 0]
            n_dynamic_obst -= 1
            coord.append([h_obs, w_obs])
    
    return coord, arr

# Function to calculate the Manhattan distance
def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

def euclidian_distance(x_st, y_st, x_end, y_end):
    return np.sqrt((x_st - x_end)**2 + (y_st - y_end)**2)
    

# Function to update the coordinates of the agent
def update_agent(agent_path, inst_arr, time_idx, local_fov, global_map, direction, agent_old_coordinates, leave_idx, dist, agent_goal, terminations, info, grid_seq, orig_cost, old_cost, expected_cell, was_reached):
    
    """ 
    Update coordinates

    Input: all paths, a map containing all information, agent id, time, local field of view size, global navigation map, 
        movement direction [x, y], coordinates at the last moment, number of grids skipped, distance

    Output: local field of view, local navigation map, global navigation map, whether to act, reward, 
        number of skipped grids, updated map, updated coordinates, distance
    """

    h, w = inst_arr.shape[:2]

    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0
    arrived = 0
    new_cost = 0

    done = False
    trunc = False
    blocked = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[0], w_old - direction[1]

    # Reward agent based on the next future step
    if expected_cell == (h_new, w_new):
        agent_reward += rewards_dict('6', +0.07)
    # else: agent_reward += rewards_dict('6', -0.01) # considering complex map simmertries this could be falsh

    # Exponential reward for close to goal behaviour 
    distance_new = euclidian_distance(h_new, w_new, agent_goal[0], agent_goal[1])
    distance_old = euclidian_distance(h_old, w_old, agent_goal[0], agent_goal[1])
    reward_new = np.exp(-1 * distance_new)

    if distance_old > distance_new:
        agent_reward += rewards_dict('6', reward_new)
    elif distance_old < distance_new:
        agent_reward += rewards_dict('6', -reward_new)
    else: 
        agent_reward += rewards_dict('6', 0)
    # Compute the reward using the exponential function

    way_to_g = (orig_cost - time_idx - 1)
    if way_to_g < 0:
        way_to_g = 0

    '''
    if (h_new, w_new) == (agent_goal[0], agent_goal[1]):
        new_cost = 0
    else:
        planner = TJPS(grid_seq[time_idx:], (h_new, w_new), (agent_goal[0], agent_goal[1]), max_wait=1, log=0)
        plan = planner.plan()
        if plan is None or plan['path_extracted'] is None:
            trunc = True 
            info['blocked'] = True
            new_cost = old_cost
        else: 
            path = plan['path_extracted']
            path = path.astype(np.int64)
            new_cost = plan['cost']

            if new_cost > old_cost:
                agent_reward += rewards_dict('6', -0.11)
            if new_cost < old_cost: 
                agent_reward += rewards_dict('6', +0.06)
            else: agent_reward += rewards_dict('6', -0.01)
            if len(path) != 1:
                expected_cell = tuple(path[1][1:3])
    #'''
        
    '''
    if(time_idx >= len(agent_path)):
        desired_pos_x = agent_path[len(agent_path) - 1][1]
        desired_pos_y = agent_path[len(agent_path) - 1][2]
    else:
        desired_pos_x = agent_path[time_idx][1]
        desired_pos_y = agent_path[time_idx][2]
    diverge = manhattan_distance(h_new, w_new, desired_pos_x, desired_pos_y)
    penalty =  diverge / (time_idx+1)
        
    # debug to monitor the agent's movement
    # print(f"At time index: {time_idx}\nAgent position: ({agent_old_coordinates[0]}, {agent_old_coordinates[1]})")
    # print(f"New agent position: ({h_new}, {w_new})\n")
    '''
    penalty = 0

    # Check if the agent has reached its goal
    if (h_new, w_new) == (agent_goal[0], agent_goal[1]):
        if was_reached == True:
            # print(f"From start position: {agent_path[0][1:3]}, agent reached it's goal at: {agent_goal} in {time_idx} timesteps, for path length: {orig_cost}\n")
            done = True
            info['goal_reached'] = True
            terminations[0] += 1
        if was_reached == False or (was_reached == True and info['goal_reached'] == True):
            # inst_arr[h_new, w_new] = [128, 0, 128]  # mark goal cell as purple
            if time_idx < orig_cost * 1.1:  # Optimal reach with small boundary
                arrived = True
                agent_reward += rewards_dict('4', orig_cost-1, time_idx)
            elif time_idx < orig_cost * 1.5:  # Close to optimal reach
                arrived = True
                agent_reward += rewards_dict('4', orig_cost-1, time_idx * 2.5)
            elif time_idx < orig_cost * 2:  # Sub-optimal reach
                # arrived = True
                agent_reward += rewards_dict('6', +0.04)
            else:  # Not optimal reach (small reward)
                agent_reward += rewards_dict('6', -0.01)
        was_reached = True
        

    # Check for out of bounds or collisions with obstacles
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0) or \
       (inst_arr[h_new, w_new][0] == 0 and inst_arr[h_new, w_new][1] == 255 and inst_arr[h_new, w_new][2] == 101) or \
       (inst_arr[h_new, w_new][0] == 0 and inst_arr[h_new, w_new][1] == 0 and inst_arr[h_new, w_new][2] == 0) or \
        (inst_arr[h_new, w_new][0] == 203 and inst_arr[h_new, w_new][1] == 0 and inst_arr[h_new, w_new][2] == 255):
        agent_reward += rewards_dict('1')
        # print("Reward for collision")
        trunc = True
        info['collision'] = True
        # print("Collision with obstacles or out of bounds")
        terminations[3] += 1
        h_new, w_new = h_old, w_old
    else:
        if direction[0] == 0 and direction[1] == 0: 
            if was_reached == True:
                agent_reward += rewards_dict('6', +0.11)
            agent_reward += rewards_dict('0')
            # print("Reward for non global navigation")
        else:
            if global_map[h_new, w_new] == 255:
                # agent_reward += rewards_dict('0')
                info['goal_reached'] = False 

                # print("Reward for non global navigation")
                # Calculate the number of non-255 cells in global_map

                # Find the index of the leave positon in the global path
                if global_map[h_old, w_old] != 255 or leave_idx == -1:
                    leave_idx = np.where((agent_path[:, 1] == h_old) & (agent_path[:, 2] == w_old))
                    leave_idx = int(leave_idx[0][0])

                    # Punish the agent upon leaving the global path
                    # agent_reward += rewards_dict('5')

            # Following the original global path  
            if global_map[h_new, w_new] != 255 and leave_idx == -1:
                # agent_reward += rewards_dict('3')
                agent_reward += rewards_dict('6', +0.03)

                # print("Reward for staying on the global path")
            
            if global_map[h_new, w_new] != 255 and leave_idx >= 0:
                # Find the index of the current position in the global path
                return_index = np.where((agent_path[:, 1] == h_new) & (agent_path[:, 2] == w_new))
                return_index = int(return_index[0][0])
                cells_skipped = return_index - leave_idx - 1
                wasted = (time_idx - way_to_g) / 10
                cell_given = cells_skipped - wasted
                if cell_given < 1:
                    cell_given = 1
                # Account for returning to the path to the next global cell is 0 cell's skipped (-1)

                # agent_reward += rewards_dict('2', cell_given)
                # print("Reward for retruning to global path")
                leave_idx = -1
                
                # Update the global_map to reflect the new global path
                for (t,x,y) in agent_path[:return_index]:
                    global_map[x,y] = 255  # or another value indicating it's no longer on the path
                    

    # Calculate new distance
    new_dist = manhattan_distance(h_new, w_new, agent_goal[0], agent_goal[1])
    if new_dist < dist:
        dist = new_dist

    # Update agent position before moving obstacles
        # Clear the previous agent position
    inst_arr[h_old, w_old] = [255, 255, 255]
    inst_arr[h_new, w_new] = [255, 0, 0]

    # Update local observation and global map
    local_obs = inst_arr[max(0, h_new - local_fov):min(h, h_new + local_fov), max(0, w_new - local_fov):min(w, w_new + local_fov)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0, h_new - local_fov):min(h, h_new + local_fov), max(0, w_new - local_fov):min(w, w_new + local_fov)]

    return np.array(local_obs), np.array(local_map), global_map, done, trunc, info, agent_reward, leave_idx, inst_arr, [h_new, w_new], dist, arrived, terminations, new_cost, penalty, expected_cell, was_reached


def rewards_dict(case, N = 0, time_idx = 1):

    """
    Return reward value
    r1 indicates that the robot reaches the free point of non-global navigation
    r2 means the robot hit an obstacle
    r3 indicates that the robot return to the global navigation path
    r4 agent follows it's global guidance path
    r5 agent reaches it's goal
    r6 agent leaves it's global path
    r7 custom configuration
    """
    r1, r2, r3, r4, r5, r6, r7 = -0.01, -0.99, 0.1, 0.2, N/time_idx, -0.03, N
    rewards = {
        '0': r1,
        '1': r1 + r2,
        '2': r1 + N * r3,
        '3': r4,
        '4': r5,
        '5': r6,
        '6': r7
    }

    return rewards[case]
