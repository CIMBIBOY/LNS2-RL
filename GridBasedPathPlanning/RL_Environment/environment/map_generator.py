from PIL import Image
import numpy as np
import matplotlib as plt 
import os

'''
This script is responsible for generating maps and converting map data. 
It includes functions to:

	1.	Create random maps with obstacles placed randomly.
	2.	Generate guide maps with specific coordinates marked.
	3.	Convert maps to value maps where obstacles are represented by 1 and free spaces by 0.
	4.	Generate start and end points for dynamic obstacles.
	5.	Create global and local guidance maps based on the paths of dynamic obstacles.
	6.	Generate heuristic values for the A* algorithm based on Manhattan distances.
'''

def random_map(w, h, n_static, max_segment_length=10, map_name="random_connected_map_patterns", color_coord=[0, 0, 0], rng=None, save_dir="GridBasedPathPlanning/RL_Environment/data/map_gen"):
    """
    Generates a random map with connected static objects (L-shaped or random patterns), ensuring connectivity and no 2x2 black regions.

    Parameters:
        w (int): Width of the map.
        h (int): Height of the map.
        n_static (int): Number of static objects (black cells).
        max_segment_length (int): Maximum length of each connected segment.
        map_name (str): Name of the map file (without extension).
        color_coord (list): RGB color for static points (walls).
        rng (numpy.random.Generator): Random number generator.
        save_dir (str): Directory to save the generated map.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure dimensions are even
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    # Initialize the map with all white cells
    data = np.ones((h, w, 3), dtype=np.uint8) * 255  # White cells
    maze = np.zeros((h, w), dtype=np.uint8)  # Binary map: 0 = path (white), 1 = wall (black)

    def can_place(x, y):
        """Check if placing a wall at (x, y) will maintain connectivity and prevent 2x2 black regions."""
        # Check 2x2 block condition
        if x > 0 and y > 0 and maze[y - 1, x - 1] == 1 and maze[y - 1, x] == 1 and maze[y, x - 1] == 1:
            return False
        if x > 0 and y < h - 1 and maze[y + 1, x - 1] == 1 and maze[y + 1, x] == 1 and maze[y, x - 1] == 1:
            return False
        if x < w - 1 and y > 0 and maze[y - 1, x + 1] == 1 and maze[y - 1, x] == 1 and maze[y, x + 1] == 1:
            return False
        if x < w - 1 and y < h - 1 and maze[y + 1, x + 1] == 1 and maze[y + 1, x] == 1 and maze[y, x + 1] == 1:
            return False

        # Temporarily place the wall
        maze[y, x] = 1

        # Check connectivity using flood fill
        visited = np.zeros_like(maze, dtype=bool)
        stack = [(0, 0)]
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < w and 0 <= cy < h and not visited[cy, cx] and maze[cy, cx] == 0:
                visited[cy, cx] = True
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

        # Check if all white cells are reachable
        reachable = np.count_nonzero(visited)
        total_white = (maze == 0).sum()

        # Remove the temporary wall
        maze[y, x] = 0

        return reachable == total_white

    def add_random_pattern(x, y, length):
        """Generate a random connected pattern starting from (x, y)."""
        directions = ["up", "down", "left", "right"]
        for _ in range(length):
            if not can_place(x, y):
                break
            maze[y, x] = 1
            direction = rng.choice(directions)
            if direction == "up" and y > 0:
                y -= 1
            elif direction == "down" and y < h - 1:
                y += 1
            elif direction == "left" and x > 0:
                x -= 1
            elif direction == "right" and x < w - 1:
                x += 1

    # Randomly generate connected patterns
    placed_static = 0
    while placed_static < n_static:
        # Start at a random position
        x, y = rng.integers(0, w), rng.integers(0, h)
        if maze[y, x] == 0:  # Start only on white cells
            pattern_length = rng.integers(2, max_segment_length + 1)
            add_random_pattern(x, y, pattern_length)
            placed_static = (maze == 1).sum()

    # Convert the binary maze to an RGB map
    data[maze == 1] = color_coord  # Black walls
    data[maze == 0] = [255, 255, 255]  # White paths

    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the map as an image
    img = Image.fromarray(data, 'RGB')
    img_path = os.path.join(save_dir, f"{map_name}.png")
    img.save(img_path)

    print(f"Map saved at: {img_path}")


def guide_map(w,h,h_coord,w_coord, map_name = "guide-1", color_coord = [50,205,50]):

    assert len(h_coord) == len(w_coord), "Coordinates length is not same"
    data = np.ones((h, w, 3), dtype=np.uint8)*255

    for i in range(len(h_coord)):
        data[h_coord[i], w_coord[i]] = color_coord
    
    img = Image.fromarray(data, 'RGB')
    img.save(f'data/{map_name}.png')


def generate_sparse_maze(w, h, map_name="sparse_maze", color_wall=[0, 0, 0], color_path=[255, 255, 255], save_dir="GridBasedPathPlanning/RL_Environment/data/map_gen"):
    """
    Generates a sparse maze using a randomized algorithm and saves it as an image.

    Parameters:
        w (int): Width of the maze (should be odd for proper maze structure).
        h (int): Height of the maze (should be odd for proper maze structure).
        map_name (str): Name of the maze file (without extension).
        color_wall (list): RGB color for maze walls.
        color_path (list): RGB color for maze paths.
        save_dir (str): Directory to save the generated maze.
    """
    # Ensure dimensions are odd
    w = w if w % 2 == 1 else w - 1
    h = h if h % 2 == 1 else h - 1

    # Initialize the maze grid (1 for walls, 0 for paths)
    maze = np.ones((h, w), dtype=np.uint8)

    # Randomized Prim's algorithm for maze generation
    def add_walls(x, y):
        """Adds walls around a given cell to the wall list."""
        if x >= 2:
            wall_list.append((x - 1, y, x - 2, y))
        if x < h - 2:
            wall_list.append((x + 1, y, x + 2, y))
        if y >= 2:
            wall_list.append((x, y - 1, x, y - 2))
        if y < w - 2:
            wall_list.append((x, y + 1, x, y + 2))

    # Start maze generation
    start_x, start_y = np.random.randint(1, h, 2) // 2 * 2 + 1  # Random odd starting point
    maze[start_x, start_y] = 0  # Mark as path
    wall_list = []
    add_walls(start_x, start_y)

    while wall_list:
        # Randomly select a wall
        idx = np.random.randint(len(wall_list))
        x1, y1, x2, y2 = wall_list.pop(idx)

        # If the cell on the opposite side of the wall is a wall, carve a path
        if maze[x2, y2] == 1:
            maze[x1, y1] = 0  # Carve through the wall
            maze[x2, y2] = 0  # Carve the target cell
            add_walls(x2, y2)

    # Convert maze to an image
    img_data = np.zeros((h, w, 3), dtype=np.uint8)
    img_data[maze == 1] = color_wall  # Walls
    img_data[maze == 0] = color_path  # Paths

    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the maze as an image
    img = Image.fromarray(img_data, 'RGB')
    img_path = os.path.join(save_dir, f"{map_name}.png")
    img.save(img_path)

    print(f"Sparse maze saved at: {img_path}")


def map_to_value(arr):
    """
    Generate an array of the map and convert the RGB values ​​into 0 and 1. 
    0 means passable and 1 means static obstacles (black)
    """

    # Eval debug map_to_value conversion call
    # print("Converting map to value array")
    
    h, w = arr.shape[:2]
    new_arr = np.zeros(shape=(h,w), dtype=np.int8)
    obstacle_count = 0
    for i in range(h):
        for j in range(w):
            cell_coord = arr[i,j]
            if cell_coord[0] == 0 and cell_coord[1] == 0 and cell_coord[2] == 0:
                new_arr[i,j] = 1
                obstacle_count += 1
    
    if np.all(new_arr == 0):
        print("Warning: All-zero value map")
    # Eval Debug static + dynamic object count along with cell num
    # print(f"Identified {obstacle_count} obstacles out of {h*w} cells")
    
    return new_arr

# Function to calculate the Manhattan distance
def manhattan_distance(start, end):
    return abs(int(end[0]) - int(start[0])) + abs(int(end[1]) - int(start[1]))

def start_end_points(obs_coords, grid, width=48, rng=None, min_manhattan_dist=20):
    """
    Generate end coordinates for dynamic obstacles based on the start coordinates.

    Input: 
    - obs_coords: coordinates of all dynamic obstacles (start coordinates)
    - grid: the grid object, which provides the map and methods like GetRandomFreeCell.
    - min_manhattan_dist: the minimum required Manhattan distance between the start and end points.

    Output: list of [dynamic obstacle id, [start point coordinates, end point coordinates]]
    """

    if rng is None:
        rng = np.random  # Use the global numpy RNG if none is provided

    coords = []

    # Loop through the start coordinates in obs_coords
    for i, start in enumerate(obs_coords):
        attempts = 0
        while attempts < 1000:
            try:
                # Generate a new random end point
                new_point = grid.GetRandomFreeCell(start, r=max(width, width))  # Use grid's method
            except RuntimeError:
                print(f"Warning: Could not find valid end point for agent {i} after {attempts} attempts")
                return None  # Return None if we can't find a valid configuration

            # Check if the Manhattan distance is greater than the minimum threshold
            if manhattan_distance(start, new_point) >= min_manhattan_dist and tuple(new_point):
                coords.append(start)  # Add start and end point as a list
                coords.append(new_point)  # Add start and end point as a list
                break  # End point is valid, break out of the while loop

            attempts += 1

        if attempts == 1000:
            print(f"Warning: Could not find valid end point for agent {i} after 1000 attempts")
            return None  # Return None if we can't find a valid configuration

    return coords

def global_guidance(path, arr):

    guidance = np.ones((len(arr), len(arr[0])), np.uint8)*255
    for i, pos in enumerate(path):
        guidance[int(pos[1]),int(pos[2])] = 105

    return guidance

def local_guidance(paths, arr, idx):
    if idx < len(paths):
        arr[paths[idx]] = [255,255,255]
        
    return arr

def heuristic_generator(arr, end):
    """
    Generate a table of heuristic function values
    """
    # print(f"Input arr shape: {arr.shape}")
    
    if len(arr.shape) == 2:
        h, w = arr.shape
    elif len(arr.shape) == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError("Invalid input array shape")
    
    # Check if end coordinates are within bounds
    if not (0 <= end[0] < h and 0 <= end[1] < w):
        raise ValueError(f"End coordinates {end} are out of bounds for the map size ({h}, {w})")

    # Initialize heuristic map with zeroes
    h_map = [[0 for _ in range(w)] for _ in range(h)]
    
    # Compute Manhattan distances
    for i in range(h):
        for j in range(w):
            h_map[i][j] = abs(end[0] - i) + abs(end[1] - j)

    return h_map

if __name__ == '__main__':
    # Map gen:
    # Example usage 
    for i in range(0, 201): 
        static_obj = 300 + i*4
        segment_length = 30 + i*4
        random_map(
            w=64,  # Width of the map
            h=64,  # Height of the map
            n_static=static_obj,  # Number of static objects
            max_segment_length=segment_length,  # Maximum length of each connected segment
            map_name=f"good{i}",
            save_dir="GridBasedPathPlanning/RL_Environment/data/map_gen"
        )

    # Example maze:
    # generate_sparse_maze(64, 64, map_name="sparse_maze_example")