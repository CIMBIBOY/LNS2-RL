import numpy as np
import heapq

class Node:
    def __init__(self, x, y, g=float('inf'), parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.parent = parent

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.g < other.g

    def __hash__(self):
        return hash((self.x, self.y))

def dijkstra(grid, start, end):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    distance = np.full((rows, cols), np.inf)
    parent = {}

    start.g = 0
    distance[start.x, start.y] = 0
    heap = [(0, start)]

    while heap:
        dist, current = heapq.heappop(heap)
        visited[current.x, current.y] = True

        if current == end:
            break

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # Possible moves including diagonals
        for dx, dy in neighbors:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                cost = current.g + (grid[nx, ny] if dx == 0 or dy == 0 else np.sqrt(2) * grid[nx, ny])
                if cost < distance[nx, ny]:
                    distance[nx, ny] = cost
                    heapq.heappush(heap, (cost, Node(nx, ny, g=cost, parent=current)))
                    parent[Node(nx, ny)] = current

    # Reconstruct path
    path = []
    node = end
    while node != start:
        path.append(node)
        node = node.parent
    path.append(start)
    path.reverse()

    return path

# Example usage:
grid = np.array([[1, 3, 1, 2],
                 [2, 2, 3, 4],
                 [3, 2, 1, 3],
                 [4, 1, 1, 2]])

start_node = Node(0, 0)
end_node = Node(3, 3)

shortest_path = dijkstra(grid, start_node, end_node)
print("Shortest Path:")
for node in shortest_path:
    print("({}, {})".format(node.x, node.y))