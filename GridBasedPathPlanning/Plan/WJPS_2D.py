import os, sys
import heapq
import numpy as np
import math
import time

from GridBasedPathPlanning.Plan.Dijkstra_mod import Node_Dijkstra, dijkstra
from GridBasedPathPlanning.Plan.Dijkstra_mod import motions as motions_Dijkstra

DEBUG = 1

class Node(object):
    def __init__(self, current: tuple, parent_motion_idx: int=None, t:int=0, g: float=0, jp_data = None) -> None:
        self.current = current
        self.parent_motion_idx = parent_motion_idx
        self.t = t
        self.g = g
        self.g_prospective = None
        self.h = None
        self.jp_data = jp_data  # Jump pont x,y,t
    
    def add(self, move, grid):  # ONLY applicable for moves, random nodes can not be added
        next_node_current = (self.x + move.x, self.y + move.y)
        
        if not np.all((np.array(next_node_current) >= 0) & (np.array(next_node_current) < np.array(grid.shape))):
            if DEBUG: print('Next node outside the grid')
            return None

        return Node(next_node_current,
                    move.parent_motion_idx,
                    t = self.t+1,
                    g = self.g + move.g * ((grid[self.current] + grid[next_node_current])/2),
                    jp_data = self.jp_data)
    
    def sub(self, node, grid): # ONLY to get previous cell, cost and time is not calculated
        prev_node_current = (self.x - node.x, self.y - node.y)
        
        if not np.all((np.array(prev_node_current) >= 0) & (np.array(prev_node_current) < np.array(grid.shape))):
            if DEBUG: print('Prev node outside the grid ???')
            return None

        return Node(prev_node_current, t = self.t-1, jp_data=self.jp_data)
    
    def __eq__(self, node) -> bool:
        return ((self.current == node.current))
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)
    def __lt__(self, node) -> bool:
        return self.g + self.h < node.g + node.h or \
              (self.g + self.h == node.g + node.h and self.h < node.h)
    def __hash__(self) -> int:
        return hash(self.current)
    def __str__(self) -> str:
        return "NODE: curr:{} par_idx:{} t:{} g:{} h:{} jp_data:{}" \
                .format(self.current, self.parent_motion_idx, self.t, self.g, self.h, self.jp_data)
    
    @property
    def x(self) -> float: return self.current[0]
    @property
    def y(self) -> float: return self.current[1]

motions = [ Node((1,0),  parent_motion_idx = 0, g = 1),
            Node((-1,0), parent_motion_idx = 1, g = 1),
            Node((0,1),  parent_motion_idx = 2, g = 1),
            Node((0,-1), parent_motion_idx = 3, g = 1),
            Node((1,1),  parent_motion_idx = 4, g = np.sqrt(2)),
            Node((-1,-1),parent_motion_idx = 5, g = np.sqrt(2)),
            Node((1,-1), parent_motion_idx = 6, g = np.sqrt(2)),
            Node((-1,1), parent_motion_idx = 7, g = np.sqrt(2))]

class WJPS:
    def __init__(self, start, goal, grid, heuristic_type: str = "octile") -> None:
        self.start = Node(start)
        self.goal = Node(goal)
        self.grid = grid
        self.grid_shape = grid.shape
        self.motions = motions
        self.heuristic_type = heuristic_type

        self.OPEN = []
        self.CLOSED = []
        self.node = None    # Node from open list, origin of jumping

    def plan(self):

        timestart = time.time()

        # OPEN set with priority and a CLOSED set
        self.OPEN = []
        self.CLOSED = []
        heapq.heappush(self.OPEN, self.start)
        
        while self.OPEN:
            self.node = heapq.heappop(self.OPEN)
            if DEBUG: print(f"NODE FROM OPEN LIST: {self.node}")

            # exists in CLOSED set
            if self.node in self.CLOSED:
                continue

            # goal found
            if self.node.current == self.goal.current:
                self.CLOSED.append(self.node)
                print("GOAL FOUND")
                self.goal.t = self.node.t # Needed by extractpath function
                print(f'Time to plan: {(time.time() - timestart)*1000} [ms]')

                cost, path = self.extractPath()
                path_extracted = self.extractPath_StepByStep(path)

                return path, path_extracted, cost

            self.get_successors(self.node)

            self.CLOSED.append(self.node)
        return [], []


    def get_successors(self, node:Node):
        possible_moves = self.get_moves(node)
        print(possible_moves)

        if DEBUG: print(f"SUCCESSOR current: {node}, direction: {self.motions[node.parent_motion_idx].current if node.parent_motion_idx is not None else None}")

        if possible_moves is not None:
            for direction_next in possible_moves:
                if DEBUG: print(f"JUMP DIRECTION: {direction_next.current}")
                self.jump(node.add(direction_next,self.grid), direction_next)
        return
    
    
    def jump(self, node:Node, move_next:Node):
        if (next_node := node.add(move_next, self.grid)) is None: return None
        if DEBUG: print(f"JUMP current: {node.current}, par_idx: {node.parent_motion_idx}, direction: {move_next.current}, next node: {next_node.current}, g: {next_node.g}")
        
        # Goal found
        if node.current == self.goal.current:
            if DEBUG: print("GOAL FOUND 1")
            self.append_OPEN_heapq(node)
            return None

        # Forced neighbour
        neighbour_grid, neighbour_index_node = self.getNeighbourGrid(next_node)
        if len(np.unique(neighbour_grid)) > 1:
            self.append_OPEN_heapq(next_node)
        
        # Diagonal move
        if all(move_next.current): # Only works for 2D !!!!!
            if move_next.parent_motion_idx == 4: 
                moves_ortho = [0,2]
            elif move_next.parent_motion_idx == 5:
                moves_ortho = [1,3]
            elif move_next.parent_motion_idx == 6:
                moves_ortho = [0,3]
            elif move_next.parent_motion_idx == 7:
                moves_ortho = [1,2]

            for i in moves_ortho:
                jumpnode = self.jump(next_node,self.motions[i])
                # if not None it is automatically added to the OPEN heap

        return self.jump(node.add(move_next,self.grid), move_next)


    def get_moves(self, node:Node):
        if node.parent_motion_idx is None: # Start Node -> All possible moves
            possible_motions = []
            for move in self.motions:
                if (next_node := node.add(move,self.grid)):
                    possible_motions.append(move)
            return possible_motions

        else:
            neighbour_grid, neighbour_index_node = self.getNeighbourGrid(node)
            neighbour_index_parent = neighbour_index_node - np.array(self.motions[node.parent_motion_idx].current)

            possible_motions = []
            for move in self.motions:

                # If on grid and not reverse movement
                if (next_node := node.add(move, self.grid)) and \
                    self.motions[node.parent_motion_idx].current != tuple(-np.array(move.current)): # NO REVERSE

                    neighbour_index_nextnode = [neighbour_index_node[0] + move.current[0],neighbour_index_node[1] + move.current[1]]
                    all_nodes = dijkstra(neighbour_grid,
                                        Node_Dijkstra(tuple(neighbour_index_parent)),
                                        motions_Dijkstra,
                                        center = neighbour_index_node)

                    if DEBUG and False: 
                        print(node, move, next_node)
                        print(neighbour_grid)
                        print(neighbour_index_parent, neighbour_index_node, neighbour_index_nextnode)
                        print(motions[all_nodes[tuple(neighbour_index_nextnode)].parent_motion_idx].current)

                    if move.current == motions[all_nodes[tuple(neighbour_index_nextnode)].parent_motion_idx].current:
                        print('same')
                        possible_motions.append(move)

        return possible_motions if len(possible_motions) else None


    def getNeighbourGrid(self, node:Node):
        neighbour_index_min = np.clip(np.array(node.current) - 1, 0, np.array(self.grid_shape) - 1)
        neighbour_index_max = np.clip(np.array(node.current) + 1, 0, np.array(self.grid_shape) - 1)
        neighbour_index_node = [] # The index of the 'node' in the exracted NeighbourGrid
        for i in range(len(self.grid_shape)):
            if node.current[i] == 0:
                neighbour_index_node.append(0)
            elif node.current[i] == self.grid_shape[i]:
                neighbour_index_node.append(-1)
            else:
                neighbour_index_node.append(1)

        neighbour_slice = tuple(slice(neighbour_index_min[i], neighbour_index_max[i]+1) for i in range(len(neighbour_index_min)))
        neighbour_grid = self.grid[neighbour_slice]
        return neighbour_grid, neighbour_index_node

    def append_OPEN_heapq(self,jp):
        if jp and jp not in self.CLOSED:

            if jp in self.OPEN:
                if jp.g < self.OPEN[self.OPEN.index(jp)].g:
                    self.OPEN.remove(jp)
                else: 
                    return None

            jp.jp_data = (self.node.t, self.node.x, self.node.y)
            jp.h = self.h(jp, self.goal,self.heuristic_type)
            heapq.heappush(self.OPEN, jp)
            if DEBUG: print(f"NODE TO OPEN LIST: {jp}")

            if jp.current == self.goal.current:
                return 1 # GOAL
        return None # Nothing is added to the queue

    def extractPath(self):

        node = self.CLOSED[self.CLOSED.index(self.goal)]

        cost = node.g
        if DEBUG: print(node)

        print("OPEN:________________________________________________________")
        for item in self.OPEN: 
            print(item)
        print()


        print("CLOSED:________________________________________________________")
        for item in self.OPEN: 
            print(item)
        print()


        path = [node]
        
        """
        while node != self.start:
            node_parent = self.OPEN[self.OPEN.index(Node(*(node.jp_data[1:]), t=node.jp_data[0]))]
            if DEBUG: print(node_parent)
            node = node_parent
            path.append(node)
        """
        print()

        while node != self.start:
            node_parent = node.sub(motions[node.parent_motion_idx],self.grid)
            
            node_parent.parent_motion_idx = node.parent_motion_idx

            if node_parent in self.OPEN:
                node_parent = self.OPEN[self.OPEN.index(node_parent)]

            print(node,motions[node.parent_motion_idx],node_parent)

            node = node_parent
            path.append(node)

        path.reverse()
        return cost, path

    def extractPath_StepByStep(self,path):
        extracted_path = [path[0].current]  # 0 th element

        for i in range(len(path)-1):
            start = path[i]
            end = path[i+1]
            while start.t < end.t:
                diff = np.array(end.current) - np.array(start.current)
                print('stuck')
                pass

        return extracted_path

    def h(self, node_1: Node, node_2: Node, heuristic_type) -> float:
        '''
        Calculate heuristic (cost)
        '''
        if heuristic_type == "manhattan":
            return abs(node_2.x - node_1.x) + abs(node_2.y - node_1.y)
        elif heuristic_type == "euclidean":
            return math.hypot(node_2.x - node_1.x, node_2.y - node_1.y)
        elif heuristic_type == "octile":
            dx = abs(node_2.x - node_1.x)
            dy = abs(node_2.y - node_1.y)
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
        else: 
            raise SyntaxError