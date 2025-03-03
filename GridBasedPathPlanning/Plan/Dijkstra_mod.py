import numpy as np
import heapq

class Node_Dijkstra():
    def __init__(self, current: tuple, parent_motion_idx=None, g=float('inf')):
        self.current = current
        self.parent_motion_idx = parent_motion_idx # parent_motion_idx (or motion_idx in case of self.Motions)
        self.g = g
    
    def __add__(self, other):
        return Node_Dijkstra((self.x + other.x, self.y + other.y), 
                             parent_motion_idx=other.parent_motion_idx,
                             g=self.g)
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

def dijkstra(grid: np.ndarray, start: Node_Dijkstra, motions: list[Node_Dijkstra], center:tuple=(1,1)):
    OPEN = []
    CLOSED = set()
    all_nodes = {}
    
    start.g = 0
    heapq.heappush(OPEN, start)
    all_nodes[start.current] = start

    while OPEN:
        node = heapq.heappop(OPEN)

        if node.current in CLOSED:
            continue

        CLOSED.add(node.current)

        for new_node in getNeighbor_Dijkstra(node, grid, grid.shape, motions, center):
            if new_node.current in CLOSED:
                continue

            if new_node.g < all_nodes.get(new_node.current, Node_Dijkstra(new_node.current)).g:
                all_nodes[new_node.current] = new_node
                heapq.heappush(OPEN, new_node)
    
    return all_nodes

def getNeighbor_Dijkstra(node: Node_Dijkstra, grid: np.ndarray, shape: tuple, motions: list[Node_Dijkstra], center:tuple) -> list:
    neighbors = []
    for motion in motions:
        new_node = node + motion
        if 0 <= new_node.x < shape[0] and 0 <= new_node.y < shape[1]:
            new_node.g = node.g + motion.g * ((grid[node.current] + grid[new_node.current])/2)
            if new_node.current == center: new_node.g = new_node.g - 1e-8 # PATH PREFERENCE TO CENTER CELL
            
            new_node.parent_motion_idx = motion.parent_motion_idx
            neighbors.append(new_node)
    return neighbors

def extractPaths_Dijkstra(all_nodes: dict, motions: list[Node_Dijkstra]):
    paths = {}
    for current, node in all_nodes.items():
        path = [node]
        while node.parent_motion_idx is not None:
            parent_motion = motions[node.parent_motion_idx]
            parent_node = all_nodes[tuple(np.array(node.current) - np.array(parent_motion.current))]
            node = parent_node
            path.append(node)
        path.reverse()
        paths[current] = path
    return paths

motions = [Node_Dijkstra((1, 0), parent_motion_idx=0, g=1),
           Node_Dijkstra((-1, 0), parent_motion_idx=1, g=1),
           Node_Dijkstra((0, 1), parent_motion_idx=2, g=1),
           Node_Dijkstra((0, -1), parent_motion_idx=3, g=1),
           Node_Dijkstra((1, 1), parent_motion_idx=4, g=np.sqrt(2)),
           Node_Dijkstra((-1, -1), parent_motion_idx=5, g=np.sqrt(2)),
           Node_Dijkstra((1, -1), parent_motion_idx=6, g=np.sqrt(2)),
           Node_Dijkstra((-1, 1), parent_motion_idx=7, g=np.sqrt(2))]

if __name__ == '__main__':
    grid = np.array([
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1., 10., 10., 10., 10., 10.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
    
    '''
    grid = np.array([[2*np.sqrt(2), 5-2*np.sqrt(2), 2*np.sqrt(2)-2],
                     [10, 2*np.sqrt(2), 5-2*np.sqrt(2)],
                     [10, 10, 2*np.sqrt(2)]])
    '''


    start_node = Node_Dijkstra((0,0))
    all_nodes = dijkstra(grid, start_node, motions, center=(1,1))
    paths = extractPaths_Dijkstra(all_nodes, motions)

    for goal, path in paths.items():
        print(f"Path to {goal}:")
        for node in path:
            print(node)
        print()


    print()
    node1 = all_nodes[(8,8)]
    print(node1)
    parent_motion = motions[node1.parent_motion_idx]
    print(parent_motion)