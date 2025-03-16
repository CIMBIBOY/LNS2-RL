import os, sys
import heapq
import numpy as np
import math

class Node(object):
    '''
    Class for searching nodes.

    Parameters
    ----------
    current: tuple
        current coordinate
    parent: tuple
        coordinate of parent node
    g: float
        path cost
    h: float
        heuristic cost
    '''
    def __init__(self, current: tuple, parent: tuple=None, g: float=0, h: float=0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, node):
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h)

    def __eq__(self, node) -> bool:
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "----------\ncurrent:{}\nparent:{}\ng:{}\nh:{}\n----------" \
            .format(self.current, self.parent, self.g, self.h)
    
    @property
    def x(self) -> float:
        return self.current[0]
    
    @property
    def y(self) -> float:
        return self.current[1]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None
        
    def shift(self, node):
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g, self.h)

class JPS:
    '''
    Class for JPS motion planning.

    Parameters
    ----------
    start (tuple): start point coordinate
    goal (tuple): goal point coordinate
    env (Env): environment
    heuristic_type (str): heuristic function type, default is euclidean
    '''
    def __init__(self, start, goal, grid, motions, heuristic_type: str = "euclidean", obs_threshold = 0.5) -> None:
        self.start = Node(tuple(start), tuple(start), 0, 0)
        self.goal = Node(tuple(goal), tuple(goal), 0, 0)
        self.grid = grid
        self.motions = motions
        self.obs_threshold = obs_threshold
        self.heuristic_type = heuristic_type

    def __str__(self) -> str:
        return "Jump Point Search(JPS)"

    def plan(self):
        '''
        JPS motion plan function.
        [1] Online Graph Pruning for Pathfinding On Grid Maps

        Return
        ----------
        path (list): planning path
        expand (list): all nodes that planner has searched
        '''
        # OPEN set with priority and CLOSED set
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = []

        while OPEN:
            node = heapq.heappop(OPEN)

            # exists in CLOSED set
            if node in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED.append(node)
                return self.extractPath(CLOSED), CLOSED

            jp_list = []
            for motion in self.motions:
                jp = self.jump(node, motion)

                # exists and not in CLOSED set
                if jp and jp not in CLOSED:
                    jp.parent = node.current
                    jp.h = self.h(jp, self.goal)
                    jp_list.append(jp)

            for jp in jp_list:
                # update OPEN set
                heapq.heappush(OPEN, jp)

                # goal found
                if jp == self.goal:
                    break
            
            CLOSED.append(node)
        return [], []

    def jump(self, node: Node, motion: Node):
        '''
        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes
        Return:
            jump_point (Node): jump point or None if searching fails
        '''

        # explore a new node
        new_node = node.shift(motion)               # Only modifies the position vector, g and h does not change
        new_node.parent = node.current              # Set the parent of the new node
        new_node.h = self.h(new_node, self.goal)                                # Cost to go
        new_node.g = node.g + motion.g * (1 + self.grid[new_node.current])  # Cost so far
        # cost of the node = cost of the parent node + cost of travel + multiplier of motion * cost of the new node

        # hit an obstacle
        if self.grid[tuple(new_node.current)] >= self.obs_threshold:
            return None

        # goal found
        if new_node == self.goal:
            return new_node

        # diagonal
        if motion.x and motion.y:
            # if exists jump point at horizontal or vertical
            x_dir = Node((motion.x, 0), None, 1, None)
            y_dir = Node((0, motion.y), None, 1, None)
            if self.jump(new_node, x_dir) or self.jump(new_node, y_dir):
                return new_node
            
        # if exists forced neighbor
        if self.detectForceNeighbor(new_node, motion):
            return new_node
        else:
            return self.jump(new_node, motion)


    def detectForceNeighbor(self, node, motion):
        '''
        Detect forced neighbor of node.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes
        Return
            flag (bool): True if current node has forced neighbor else Flase
        '''

        x, y = node.current
        x_dir, y_dir = motion.current
        
        # horizontal
        if x_dir and not y_dir:
            if  self.grid[x, y + 1] >= self.obs_threshold and \
                self.grid[x + x_dir, y + 1] < self.obs_threshold:
                return True
            
            if  self.grid[x, y - 1] >= self.obs_threshold and \
                self.grid[x + x_dir, y - 1] < self.obs_threshold:
                return True
        
        # vertical
        if not x_dir and y_dir:
            if  self.grid[x + 1, y] >= self.obs_threshold and \
                self.grid[x + 1, y + y_dir] < self.obs_threshold:
                return True
            
            if  self.grid[x - 1, y] >= self.obs_threshold and \
                self.grid[x - 1, y + y_dir] < self.obs_threshold:
                return True
        
        # diagonal
        if x_dir and y_dir:
            if  self.grid[x - x_dir, y] >= self.obs_threshold and \
                self.grid[x - x_dir, y + y_dir] < self.obs_threshold:
                return True
            
            if  self.grid[x, y - y_dir] >= self.obs_threshold and \
                self.grid[x + x_dir, y - y_dir] < self.obs_threshold:
                return True

        return False
    
    def extractPath(self, closed_set):
        '''
        Extract the path based on the CLOSED set.

        Parameters
        ----------
        closed_set: (list): CLOSED set

        Return
        ----------
        cost: (float) the cost of planning path
        path: (list): the planning path
        '''
        cost = 0
        node = closed_set[closed_set.index(self.goal)]
        path = [node.current]
        while node != self.start:
            node_parent = closed_set[closed_set.index(Node(node.parent, None, None, None))]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path
    
    def getNeighbor(self, node: Node) -> list:
        '''
        Find neighbors of node.

        Parameters
        ----------
        node: (Node): current node

        Return
        ----------
        neighbors: (list): neighbors of current node
        '''
        return [node + motion for motion in self.motions
                if not self.isCollision(node, node + motion)]

    def h(self, node: Node, goal: Node) -> float:
        '''
        Calculate heuristic.

        Parameters
        ----------
        node (Node): current node
        goal (Node): goal node

        Return
        ----------
        h (float): heuristic function value of node
        '''

        if self.heuristic_type == "manhattan":
            return abs(goal.x - node.x) + abs(goal.y - node.y)
        elif self.heuristic_type == "euclidean":
            return math.hypot(goal.x - node.x, goal.y - node.y)

    def cost(self, node1: Node, node2: Node) -> float:
        '''
        Calculate cost for this motion.
        '''
        if self.isCollision(node1, node2):
            return float("inf")
        return self.dist(node1, node2)
    
    def isCollision(self, node1: Node, node2: Node) -> bool:
        '''
        Judge collision when moving from node1 to node2.

        Parameters
        ----------
        node1, node2: Node

        Return
        ----------
        collision: bool
            True if collision exists else False
        '''
        if node1.current in self.obstacles or node2.current in self.obstacles:
            return True

        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        if x1 != x2 and y1 != y2:
            if x2 - x1 == y1 - y2:
                s1 = (min(x1, x2), min(y1, y2))
                s2 = (max(x1, x2), max(y1, y2))
            else:
                s1 = (min(x1, x2), max(y1, y2))
                s2 = (max(x1, x2), min(y1, y2))
            if s1 in self.obstacles or s2 in self.obstacles:
                return True
        return False
    
    def dist(self, node1: Node, node2: Node) -> float:
        return math.hypot(node2.x - node1.x, node2.y - node1.y)
    
    def angle(self, node1: Node, node2: Node) -> float:
        return math.atan2(node2.y - node1.y, node2.x - node1.x)
    