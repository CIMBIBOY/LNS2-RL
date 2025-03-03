import numpy as np
import heapq

class Node_Dijkstra():
    def __init__(self, current: tuple, parent_motion_idx = None, g=float('inf')):
        self.current = current
        self.parent_motion_idx = parent_motion_idx # parent_motion_idx (or motion_idx in case of self.Motions)
        self.g = g
    
    def __add__(self, other):
        return Node_Dijkstra((self.x + other.x, self.y + other.y), 
                             parent_motion_idx = other.parent_motion_idx,
                             g = self.g) # g is not modified !!!
    
    def __sub__(self, other):
        return Node_Dijkstra((self.x - other.x, self.y - other.y)) # Only use to get parent Node.current

    def __eq__(self, other):
        return self.current == other.current

    def __lt__(self, other):
        return self.g < other.g

    def __hash__(self):
        return hash(self.current)
    
    def __str__(self) -> str:
        return f"NODE: curr:{self.current} par_idx:{self.parent_motion_idx} g:{self.g}"

    @property
    def x(self): return self.current[0]
    
    @property
    def y(self): return self.current[1]


def dijkstra(grid, start, goal, motions):

    OPEN = []
    heapq.heappush(OPEN, start)
    CLOSED = []

    while OPEN:
        node = heapq.heappop(OPEN)

        # exists in CLOSED set
        if node in CLOSED:
            continue

        # goal found
        if node == goal:
            CLOSED.append(node)
            return extractPath_Dijkstra(CLOSED, start, goal, motions)

        for new_node in getNeighbor_Dijkstra(node, grid.shape):
            
            # hit the obstacle
            #if new_node.current in self.obstacles:
            #    continue
            
            if new_node in CLOSED:
                continue

            # goal found
            if new_node == goal:
                heapq.heappush(OPEN, new_node)
                break
            
            # update OPEN set
            heapq.heappush(OPEN, new_node)
        
        CLOSED.append(node)
    return ([], []), []

def getNeighbor_Dijkstra(node: Node_Dijkstra, shape: np.ndarray) -> list:
    neighbors = []
    for motion in motions:
        new_node = node + motion
        if 0 <= new_node.x < shape[0] and 0 <= new_node.y < shape[1]:
            new_node.g = g_Disjkstra(grid, node, motion, new_node)
            neighbors.append(new_node)
    return neighbors

def extractPath_Dijkstra(closed_set, start, goal, motions):
    node = closed_set[closed_set.index(goal)]
    path = [node]
    while node != start:
        node_parent = closed_set[closed_set.index(node - motions[node.parent_motion_idx])]
        node = node_parent
        path.append(node)
    path.reverse()
    cost = goal.g
    return cost, path

def g_Disjkstra(grid, node, move, new_node):
    return node.g + move.g * (grid[node.current] + grid[(new_node).current])/2 # average current and dest cells weights and mult with path len


if __name__ == '__main__':

    grid = np.array([[1, 3, 1, 2],
                     [2, 1, 8, 4],
                     [1, 7, 8, 2],
                     [5, 4, 1, 2]])

    motions = [ Node_Dijkstra((1,0),  parent_motion_idx = 0, g = 1),
                Node_Dijkstra((-1,0), parent_motion_idx = 1, g = 1),
                Node_Dijkstra((0,1),  parent_motion_idx = 2, g = 1),
                Node_Dijkstra((0,-1), parent_motion_idx = 3, g = 1),
                Node_Dijkstra((1,1),  parent_motion_idx = 4, g = np.sqrt(2)),
                Node_Dijkstra((-1,-1),parent_motion_idx = 5, g = np.sqrt(2)),
                Node_Dijkstra((1,-1), parent_motion_idx = 6, g = np.sqrt(2)),
                Node_Dijkstra((-1,1), parent_motion_idx = 7, g = np.sqrt(2))]


    start_node = Node_Dijkstra((0,0), g = 0)
    goal_node  = Node_Dijkstra((3,3))

    cost, path = dijkstra(grid, start_node, goal_node, motions)

    #for node in path: print(node)
    #print("Last motion: ", motions[path[-1].parent_motion_idx].current)


"""
OPEN = []
n1 = Node_Dijkstra((0,0),g=2)
n2 = Node_Dijkstra((0,0),g=1)
heapq.heappush(OPEN, n1)
print(n2 in OPEN)
heapq.heappush(OPEN, n2)

print(heapq.heappop(OPEN))
print(heapq.heappop(OPEN))
"""