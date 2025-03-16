import os, sys
import heapq
import numpy as np
import time

from GridBasedPathPlanning.Plan.Node import Node_TJPS
from GridBasedPathPlanning.Plan.motions import *

class TAStar:
    def __init__(self, grid_seq:np.ndarray, start:tuple, goal:tuple, 
                 goal_threshold:int=0,
                 diag:bool = False,
                 log:int = 1,  # 0: None, 1: Basic info, 2: DEBUG
                 max_execution_time:float = 100):
        self.start = Node_TJPS(start)
        self.goal = Node_TJPS(goal)
        self.grid_seq = grid_seq
        self.grid_shape = np.array(grid_seq[0].shape)
        self.obs_threshold = np.inf
        self.goal_threshold = goal_threshold
        self.max_execution_time = max_execution_time

        # Determine the motion set and heuristy type based on problem dimension and if diag stepping is enabled
        if len(start) == 2: self.motions = motions_2D if diag else motions_yxw
        else:               self.motions = motions_3D if diag else motions_xyzw
        self.heuristic_type = "euclidean" if diag else "manhattan"

        self. log = log

    def plan(self):
        if self.grid_seq[0,*self.start.current] >= self.obs_threshold or \
           np.all(self.grid_seq[:,*self.goal.current] >= self.obs_threshold):
            print("START or GOAL is an obstacle !! Path can not be found !!")
            return None

        time_start = time.time()

        self.OPEN = []      # OPEN set with priority
        self.CLOSED = set()    # CLOSED set
        heapq.heappush(self.OPEN, self.start)
        
        while self.OPEN:
            self.node = heapq.heappop(self.OPEN)
            if self.log == 2: print(f"NODE FROM OPEN LIST: {self.node}")

            # Check if node is in the CLOSED set
            if self.node in self.CLOSED:
                continue

            # goal found
            if self.is_goal(self.node):
                self.goal = self.node # MODIFY THE GOAL TO THE FINAL ONE
                self.CLOSED.add(self.node)
                if self.log > 0: print("GOAL FOUND: ",self.goal.current)

                path, path_extracted, cost = self.extractPath(self.CLOSED)

                time_to_plan = time.time() - time_start
                if self.log > 0: print(f'Time to plan: {time_to_plan*1000} [ms]')

                return {#'path_ori':path,
                        'path_extracted':path_extracted,
                        'cost':cost,
                        'time_to_plan':time_to_plan,
                        'grid_shape':self.grid_shape}

            self.expand_neighbors(self.node)

            self.CLOSED.add(self.node)

            if time.time() - time_start >= self.max_execution_time:
                 print('Maximum execution time exceeded')
                 break
        return None # Planning failed

    def expand_neighbors(self, node: Node_TJPS):
        """ Generate and evaluate all possible neighboring nodes. """
        for motion in self.motions:
            new_node = node + motion
            
            # Ignore node if it's already in CLOSED set
            if new_node in self.CLOSED:
                continue

            # Ensure the new node is within grid bounds and is not an obstacle
            if np.all(np.logical_and(0 <= np.array(new_node.current), new_node.current < self.grid_shape)) and \
               self.grid_seq[new_node.t, *new_node.current] < self.obs_threshold:
                # Calculate the cost and heuristic
                new_node.g = new_node.g - motion.g # cost calculated later with another cost model
                new_node.g = new_node.g + ((self.grid_seq[node.t,*node.current] + self.grid_seq[new_node.t,*new_node.current])/2 + 1) * motion.g
                new_node.h = self.h(new_node, self.goal, self.heuristic_type)

                # Add the new node to the OPEN set
                heapq.heappush(self.OPEN, new_node)

    def extractPath(self, closed_set):
        # Extract the path from the CLOSED set.

        node = next((n for n in closed_set if n == self.goal), None)
        cost = node.g
        path = [node]

        while node != self.start:
            # Trace back from the goal to the start
            node_parent_current = tuple(np.subtract(node.current, self.motions[node.parent_motion_idx].current))
            node_parent = next((n for n in closed_set if n == Node_TJPS(node_parent_current, t=node.t-1)), None)   #closed_set[closed_set.index(Node_TJPS(node_parent_current, t=node.t-1))]
            node = node_parent
            path.append(node)

        path.reverse()

        # The returned path_extracted is a numpy array with a shape of time*4, and the list of elements is: t,y,x
        path_extracted = np.empty((path[-1].t+1, len(path[0].current)+1))
        for i in range(len(path)):
            path_extracted[i] = np.array([i, *path[i].current])

        return path, path_extracted, cost

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