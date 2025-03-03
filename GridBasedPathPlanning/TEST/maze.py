import numpy as np
import random

def generate_maze(shape, density=0.3):
    """
    Generate a 2D maze represented as a numpy array.
    
    Parameters:
    - shape (tuple): Shape of the maze, i.e., (height, width)
    - density (float): The density of walls in the maze. 0 means no walls, 1 means full walls.

    Returns:
    - maze (numpy array): A 2D numpy array where 0 represents a free space, and 1 represents a wall.
    """
    height, width = shape
    
    # Ensure the dimensions are odd (to have proper walls and paths)
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1
    
    # Start with a grid full of walls
    maze = np.ones((height, width), dtype=np.int8)
    
    # Randomized DFS algorithm to generate the maze
    def dfs(x, y):
        # Define possible movements (up, down, left, right)
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)  # Randomize the directions to create randomness

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < height-1 and 1 <= ny < width-1 and maze[nx, ny] == 1:
                # If the new position is a wall, carve a path
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0  # Carve the wall between the current cell and new cell
                dfs(nx, ny)

    # Start carving the maze from a random odd position
    start_x, start_y = (random.randrange(1, height, 2), random.randrange(1, width, 2))
    maze[start_x, start_y] = 0  # The starting point
    dfs(start_x, start_y)
    
    # Adjust the density: Add more walls based on the density parameter
    wall_count = int(density * height * width)
    for _ in range(wall_count):
        wx, wy = random.randrange(1, height, 2), random.randrange(1, width, 2)
        maze[wx, wy] = 1  # Add a wall at a random location
    
    return maze


import matplotlib.pyplot as plt

maze = generate_maze((30, 30), density=0.005)

# Visualize the generated maze
plt.imshow(maze, cmap="binary")
plt.show()
