import numpy as np
import random
from collections import deque

def generate_connected_clustered_grid(shape, density, cluster_size=3):
    """
    Generate a grid with random clustered walls and structures, ensuring all open cells are connected.
    
    Parameters:
    - shape: tuple (height, width) representing the grid dimensions.
    - density: float between 0 and 1, representing the percentage of grid cells that are walls/obstacles.
    - cluster_size: int, the approximate size of wall clusters.
    
    Returns:
    - grid: A NumPy array with clustered walls (1) and open spaces (0).
    """
    height, width = shape
    grid = np.zeros((height, width), dtype=int)
    
    # Determine the number of cells to fill with walls based on the density
    num_walls = int(density * height * width)
    
    walls_placed = 0
    attempts = 0
    max_attempts = num_walls * 10  # To prevent infinite loops
    
    while walls_placed < num_walls and attempts < max_attempts:
        attempts += 1
        
        # Pick a random starting point for a new cluster
        start_row = random.randint(0, height - 1)
        start_col = random.randint(0, width - 1)
        
        # If the starting cell is already a wall, skip
        if grid[start_row, start_col] == 1:
            continue
        
        # Perform a random walk to place a cluster of walls
        current_row, current_col = start_row, start_col
        for _ in range(cluster_size):
            if walls_placed >= num_walls:
                break
            if grid[current_row, current_col] == 0:
                # Temporarily place a wall
                grid[current_row, current_col] = 1
                walls_placed += 1
                
                # Check connectivity
                if not is_grid_fully_connected(grid):
                    # If disconnected, remove the wall
                    grid[current_row, current_col] = 0
                    walls_placed -= 1
                    break  # Stop growing this cluster
            
            # Randomly move to a neighboring cell
            direction = random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up' and current_row > 0:
                current_row -= 1
            elif direction == 'down' and current_row < height - 1:
                current_row += 1
            elif direction == 'left' and current_col > 0:
                current_col -= 1
            elif direction == 'right' and current_col < width - 1:
                current_col += 1
            else:
                # If movement is out of bounds, pick another direction
                continue
    
    if walls_placed < num_walls:
        print(f"Warning: Only placed {walls_placed} walls out of {num_walls} requested.")
    
    return grid

def is_grid_fully_connected(grid):
    """
    Check if all open cells in the grid are connected.
    """
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    # Find the first open cell
    try:
        start = np.argwhere(grid == 0)[0]
    except IndexError: # No open cells
        return False
    
    queue = deque()
    queue.append(tuple(start))
    visited[start[0], start[1]] = True
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-directional
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if grid[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    
    # Check if all open cells are visited
    return np.all((grid == 1) | visited)

# Example usage:
if __name__ == "__main__":
    dim = 100
    density = 0.2     # 30% of the grid will be walls
    cluster_size = int(dim/5)  # Walls will appear in clusters of around 5 cells
    
    grid = generate_connected_clustered_grid((dim,dim), density, cluster_size)
    
    import matplotlib.pyplot as plt
    plt.imshow(grid, cmap='Greys', origin='upper')
    plt.show()
