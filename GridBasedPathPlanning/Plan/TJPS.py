import os, sys
import heapq
import numpy as np
import time

from GridBasedPathPlanning.Plan.Node import Node_TJPS
from GridBasedPathPlanning.Plan.motions import motions_yxw, motions_xyzw

class TJPS:
    """
    Args::

        grid_seq:np.ndarray,
        start:tuple,
        goal:tuple,
        obs_threshold:float=0.01, 
        goal_threshold:int=0,
        max_wait:int=0,
        log:int=1, # 0: None, 1: Basic info, 2: DEBUG
        max_execution_time:float=100
    
    :param int param1: desc
    """

    def __new__(cls, grid_seq, *args, **kwargs):
        # Check grid dimensions and return appropriate subclass instance
        if   len(grid_seq[0].shape) == 2:
            instance = super(TJPS, TJPS_2D).__new__(TJPS_2D)
        elif len(grid_seq[0].shape) == 3:
            instance = super(TJPS, TJPS_3D).__new__(TJPS_3D)
        else:
            raise ValueError("Unsupported grid dimensions. Only 2D and 3D are supported.")
        return instance

    def __init__(self,
                 grid_seq:np.ndarray,
                 start:tuple,
                 goal:tuple,
                 obs_threshold:float=0.01, 
                 goal_threshold:int=0,
                 max_wait:int=0,
                 log:int=1, # 0: None, 1: Basic info, 2: DEBUG
                 max_execution_time:float=100):
        self.start = Node_TJPS(tuple(start))
        self.goal = Node_TJPS(tuple(goal))
        self.grid_seq = grid_seq
        self.grid_shape = np.array(grid_seq[0].shape)

        self.obs_threshold = obs_threshold
        self.goal_threshold = goal_threshold
        self.max_wait = max_wait
        self.heuristic_type = "manhattan"

        self.time_index_obstacle = 0
        self.log = log
        self.max_execution_time = max_execution_time


    def plan(self):
        if not np.all(np.logical_and(0 <= np.array(self.start.current), np.array(self.start.current) < self.grid_shape)) or \
           not np.all(np.logical_and(0 <= np.array(self.goal.current),  np.array(self.goal.current)  < self.grid_shape)) or \
           self.grid_seq[0,*self.start.current] >= self.obs_threshold or \
           np.all(self.grid_seq[:,*self.goal.current] >= self.obs_threshold):
                print("START or GOAL is an obstacle or out of grid !! Path can not be found !!")
                return None

        time_start = time.time()

        self.OPEN = []      # OPEN set with priority
        self.CLOSED = set()    # CLOSED set
        heapq.heappush(self.OPEN, self.start)
        
        while self.OPEN:
            self.node = heapq.heappop(self.OPEN) # Node from open list, origin of jumping
            if self.log == 2: print(f"NODE FROM OPEN LIST: {self.node}")

            # exists in CLOSED set
            if self.node in self.CLOSED:
                continue

            # goal found
            if self.is_goal(self.node):
                self.goal = self.node # MODIFY THE GOAL TO THE FINAL ONE
                self.CLOSED.add(self.node)
                if self.log > 0: print("GOAL FOUND: ",self.goal.current)
                self.goal.t = self.node.t # Needed by extractpath function

                time_to_plan = time.time() - time_start
                if self.log > 0: print(f'Time to plan: {time_to_plan*1000} [ms]')

                path, cost = self.extractPath(self.CLOSED)
                path_extracted = self.extractPath_StepByStep(path)

                return {#'path_ori':path,
                        'path_extracted':path_extracted,
                        'cost':cost,
                        'time_to_plan':time_to_plan,
                        'grid_shape':self.grid_shape}

            self.get_successors(self.node)

            self.CLOSED.add(self.node)

            if time.time() - time_start >= self.max_execution_time:
                 print('Maximum execution time exceeded')
                 break
        print('No path found')
        return None # Planning failed

    def get_successors(self, node:Node_TJPS):
        canonical_moves, forced = self.get_canonical_moves(node)
        if canonical_moves is None:
            return

        if self.log == 2: print(f"SUCCESSOR current: {node}, direction: {self.motions[node.parent_motion_idx].current if node.parent_motion_idx is not None else None}")

        for direction_next in canonical_moves:
            if self.log == 2: print(f"JUMP DIRECTION: {direction_next.current}")

            next_node = node + direction_next
            if next_node in self.CLOSED:
                continue

            self.time_index_obstacle = node.t
            if len(node.current) == 3:
                self.jump_start_z = node.z

            self.jump(next_node)
        return

    def append_OPEN_heapq(self,jp):
        if jp and jp not in self.CLOSED:
            jp.jp_data = (self.node.t, self.node.current)
            jp.h = self.h(jp, self.goal, self.heuristic_type)
            heapq.heappush(self.OPEN, jp)
            if self.log == 2: print(f"NODE TO OPEN LIST: {jp}")

    def h(self, node_1: Node_TJPS, node_2: Node_TJPS, heuristic_type) -> float:
        if heuristic_type == "manhattan":
            return np.sum(np.abs(np.subtract(node_2.current, node_1.current)))
        elif heuristic_type == "euclidean":
            return np.linalg.norm(np.subtract(node_2.current, node_1.current))

    def is_goal(self, node: Node_TJPS):
        if self.goal_threshold == 0: 
            return node.current == self.goal.current
        else:
            return np.linalg.norm(np.subtract(self.goal.current, node.current)) <= self.goal_threshold

    def extractPath(self, closed_set):
        ''' Extract the path based on the CLOSED set. '''
        node = next((n for n in closed_set if n == self.goal), None)
        cost = node.g
        if self.log == 2: print(node)

        path = [node]
        while node != self.start:
            node_parent = next((n for n in closed_set if n == Node_TJPS(node.jp_data[1], t=node.jp_data[0])), None)
            if self.log == 2: print(node_parent)
            if self.grid_seq[node_parent.t, *node_parent.current] >= self.obs_threshold: print(f'COLLISION: {node_parent}')

            node = node_parent
            path.append(node)
        path.reverse()
        return path, cost
    
    def extractPath_StepByStep(self,path):
        # Construct a VHW(2D) or XYZW(3D) canonical step by step path from the list of temporal jump points
        # The returned path_extracted is a np array with the list of elements is: t,x,y,(z)

        extracted_path = np.empty((path[-1].t + 1, len(path[0].current)+1))
        time = 0
        extracted_path[time] = np.array([time, *path[0].current])  # 0 th element

        for i in range(len(path)-1):
            start = path[i]
            end = path[i+1]
            while start.t < end.t:
                diff = np.subtract(end.current,start.current)
                time += 1

                if len(start.current) == 2: # TJPS_2D

                    if diff[1] != 0: # Y
                        motion = Node_TJPS((0, np.sign(diff[1])))
                        start = start + motion
                    elif diff[0] != 0: # X
                        motion = Node_TJPS((np.sign(diff[0]), 0))
                        start = start + motion
                    else:
                        start.t += 1 # T

                else: # TJPS_3D

                    if diff[0] != 0: # X
                        motion = Node_TJPS((np.sign(diff[0]), 0, 0))
                        start = start + motion
                    elif diff[1] != 0: # Y
                        motion = Node_TJPS((0, np.sign(diff[1]), 0))
                        start = start + motion
                    elif diff[2] != 0: # Z
                        motion = Node_TJPS((0, 0, np.sign(diff[2])))
                        start = start + motion
                    else:
                        start.t += 1 # T


                extracted_path[time] = np.array([time, *start.current])
        return extracted_path
 

class TJPS_2D(TJPS):
    def __init__(self, grid_seq, *args, **kwargs):
        super().__init__(grid_seq, *args, **kwargs)
        self.motions = motions_yxw

    def jump(self, node:Node_TJPS):
        if self.log == 2: print(f"JUMP: {node}, move_p: {self.motions[node.parent_motion_idx].current}")

        motions_n, forced = self.get_canonical_moves(node)

        if motions_n is None:
            return None

        if self.is_goal(node):
            self.append_OPEN_heapq(node)
            return None

        # Check if a move_next exists in motions_n such that move_next is forced
        if forced:
            if self.log == 2: print(f"FORCED neighbour / temporal jump point detected: {node}")
            self.append_OPEN_heapq(node)
            return None
        
        if self.motions[-1] in motions_n: # WAIT
            max_time_index_of_disappearing_obs  = self.time_index_obstacle + self.max_wait
            if node.t < max_time_index_of_disappearing_obs:
                jump_point = self.jump(node + self.motions[-1]) # WAIT
                self.append_OPEN_heapq(jump_point)
        
        for move_n in motions_n:
            if move_n.x and not move_n.y: # HORIZONTAL MOTION
                jump_point = self.jump(node + move_n)
                self.append_OPEN_heapq(jump_point)

        for move_n in motions_n:
            if not move_n.x and move_n.y: # VERTICAL MOTION
                jump_point = self.jump(node + move_n)
                self.append_OPEN_heapq(jump_point)

        return None

    def get_canonical_moves(self, node:Node_TJPS):
        """ This function implements the TJPS Prunning rules: VHW canonical paths"""

        forced = False

        if   node.parent_motion_idx == 0: # VERTICAL + direction
            move_idx = [0,2,3,4] # every type of action is possible except the reverse action
        elif node.parent_motion_idx == 1: # VERTICAL - direction
            move_idx = [1,2,3,4] # every type of action is possible except the reverse action

        elif node.parent_motion_idx == 2: # HORIZONTAL + direction
            move_idx = [2,4] # only horizontal or wait action except the reverse action
            # FORCED TO BE GENERATED
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x-1, node.y-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x,   node.y-1] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x-1, node.y+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x,   node.y+1] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
        elif node.parent_motion_idx == 3: # HORIZONTAL - direction
            move_idx = [3,4] # only horizontal or wait action except the reverse action
            # FORCED TO BE GENERATED
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x+1, node.y-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x,   node.y-1] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x+1, node.y+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x,   node.y+1] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True

        elif node.parent_motion_idx == 4: # WAIT ACTION
            move_idx = [4]
            # If dynamic obstacle will be clearing in the next step accept an action another than wait
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t,     node.x+1, node.y] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x+1, node.y] < self.obs_threshold:
                    move_idx.append(2)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t,     node.x, node.y+1] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y+1] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t,     node.x-1, node.y] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x-1, node.y] < self.obs_threshold:
                    move_idx.append(3)
                    forced = True
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t,     node.x, node.y-1] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y-1] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
        elif node.parent_motion_idx is None:    # START CELL ONLY
            move_idx = [0,1,2,3,4]

        # CHECK WHETHER THE CANONICAL ACTIONS ARE CLEAR PATHS
        moves_checked = []
        for idx in move_idx:
            next_node = node + self.motions[idx]
            if  0 <= next_node.x < self.grid_shape[0] and \
                0 <= next_node.y < self.grid_shape[1]:
                    if self.grid_seq[next_node.t, *next_node.current] < self.obs_threshold:
                        moves_checked.append(self.motions[idx])

        return (moves_checked, forced) if len(moves_checked) else (None, forced)

class TJPS_3D(TJPS):
    def __init__(self, grid_seq, *args,
                 jump_limit_z:int=None, **kwargs):
        super().__init__(grid_seq, *args, **kwargs)
        self.motions = motions_xyzw
        self.jump_start_z = 0
        self.jump_limit_z = jump_limit_z

    def jump(self, node:Node_TJPS):
        if self.log == 2: print(f"JUMP: {node}, move_p: {self.motions[node.parent_motion_idx].current}")

        motions_n, forced = self.get_canonical_moves(node)

        if motions_n is None:
            return None

        if self.is_goal(node):
            self.append_OPEN_heapq(node)
            return None

        # Check if a move_n exists in motions_n such that move_n is forced
        if forced:
            if self.log == 2: print(f"FORCED neighbour / temporal jump point detected: {node}")
            self.append_OPEN_heapq(node)
            return None
        
        if self.motions[-1] in motions_n: # WAIT
            max_time_index_of_disappearing_obs = self.time_index_obstacle + self.max_wait
            if node.t < max_time_index_of_disappearing_obs:
                jump_point = self.jump(node + self.motions[-1]) # WAIT
                self.append_OPEN_heapq(jump_point)
        
        if (self.jump_limit_z is None) or (self.jump_limit_z and (self.jump_start_z-self.jump_limit_z <= node.z <= self.jump_start_z+self.jump_limit_z)):
            for move_n in motions_n:
                if not move_n.x and not move_n.y and move_n.z: # Z direction
                    jump_point = self.jump(node + move_n)
                    self.append_OPEN_heapq(jump_point)
            if self.jump_limit_z and (node.z == self.jump_start_z-self.jump_limit_z or node.z == self.jump_start_z+self.jump_limit_z):
                self.append_OPEN_heapq(node)

        for move_n in motions_n:
            if not move_n.x and move_n.y and not move_n.z: # Y direction
                jump_point = self.jump(node + move_n)
                self.append_OPEN_heapq(jump_point)

        for move_n in motions_n:
            if move_n.x and not move_n.y and not move_n.z: # X direction
                jump_point = self.jump(node + move_n)
                self.append_OPEN_heapq(jump_point)

        return None

    def get_canonical_moves(self, node:Node_TJPS):
        """ This function implements the TJPS Prunning rules: XYZW canonical path """

        forced = False

        if   node.parent_motion_idx == 0: # X+ direction
            move_idx = [0,2,3,4,5,6] # every type of action is possible except the reverse action

        elif node.parent_motion_idx == 1: # X- direction
            move_idx = [1,2,3,4,5,6] # every type of action is possible except the reverse action

        elif node.parent_motion_idx == 2: # Y+
            move_idx = [2,4,5,6]
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x+1, node.y-1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x+1, node.y,   node.z] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x-1, node.y-1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x-1, node.y,   node.z] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
        
        elif node.parent_motion_idx == 3: # Y-
            move_idx = [3,4,5,6]
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x+1, node.y+1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x+1, node.y,   node.z] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x-1, node.y+1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x-1, node.y,   node.z] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True

        elif node.parent_motion_idx == 4: # Z+
            move_idx = [4,6]
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x+1, node.y, node.z-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x+1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x-1, node.y, node.z-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x-1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x, node.y+1, node.z-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x, node.y+1, node.z] < self.obs_threshold:
                    move_idx.append(2)
                    forced = True
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x, node.y-1, node.z-1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x, node.y-1, node.z] < self.obs_threshold:
                    move_idx.append(3)
                    forced = True

        elif node.parent_motion_idx == 5: # Z-
            move_idx = [5,6]
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x+1, node.y, node.z+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x+1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t, node.x-1, node.y, node.z+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x-1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x, node.y+1, node.z+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x, node.y+1, node.z] < self.obs_threshold:
                    move_idx.append(2)
                    forced = True
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t, node.x, node.y-1, node.z+1] >= self.obs_threshold and \
                self.grid_seq[node.t, node.x, node.y-1, node.z] < self.obs_threshold:
                    move_idx.append(3)
                    forced = True
            
        elif node.parent_motion_idx == 6: # WAIT ACTION
            move_idx = [6]
            # If dynamic obstacle will be clearing in the next step accept an action another than wait
            # FORCED TO BE GENERATED
            if  0 <= node.x+1 < self.grid_shape[0] and \
                self.grid_seq[node.t,     node.x+1, node.y, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x+1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(0)
                    forced = True
            if  0 <= node.x-1 < self.grid_shape[0] and \
                self.grid_seq[node.t,     node.x-1, node.y, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x-1, node.y, node.z] < self.obs_threshold:
                    move_idx.append(1)
                    forced = True
            if  0 <= node.y+1 < self.grid_shape[1] and \
                self.grid_seq[node.t,     node.x, node.y+1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y+1, node.z] < self.obs_threshold:
                    move_idx.append(2)
                    forced = True
            if  0 <= node.y-1 < self.grid_shape[1] and \
                self.grid_seq[node.t,     node.x, node.y-1, node.z] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y-1, node.z] < self.obs_threshold:
                    move_idx.append(3)
                    forced = True
            if  0 <= node.z+1 < self.grid_shape[2] and \
                self.grid_seq[node.t,     node.x, node.y, node.z+1] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y, node.z+1] < self.obs_threshold:
                    move_idx.append(4)
                    forced = True
            if  0 <= node.z-1 < self.grid_shape[2] and \
                self.grid_seq[node.t,     node.x, node.y, node.z-1] >= self.obs_threshold and \
                self.grid_seq[node.t + 1, node.x, node.y, node.z-1] < self.obs_threshold:
                    move_idx.append(5)
                    forced = True
        elif node.parent_motion_idx is None:    # START CELL ONLY
             move_idx = [0,1,2,3,4,5,6]

        # CHECK WHETHER THE CANONICAL ACTIONS ARE CLEAR PATHS
        moves_checked = []
        for idx in move_idx:
            next_node = node + self.motions[idx]
            if  0 <= next_node.x < self.grid_shape[0] and \
                0 <= next_node.y < self.grid_shape[1] and \
                0 <= next_node.z < self.grid_shape[2]:
                    if self.grid_seq[next_node.t, *next_node.current] < self.obs_threshold:
                        moves_checked.append(self.motions[idx])

        return (moves_checked, forced) if len(moves_checked) else (None, forced)
    