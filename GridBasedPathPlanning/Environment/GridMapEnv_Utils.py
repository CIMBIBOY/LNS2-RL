import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO
import itertools
from noise import pnoise3, snoise2
import pickle
import colorsys


def getBenchmarkMap(map_name, reduction:int=1):
    """ Berlin_0_256.map, lak303d.map """

    with open("GridBasedPathPlanning/Data/BenchmarkMaps/" + map_name) as file:
        file.readline()
        y_size = int((int)(file.readline()[7:10])/reduction)
        x_size = int((int)(file.readline()[6:9])/reduction)
        file.readline()
        map_str = file.read()
        map_str = map_str.replace(".","0 ")
        map_str = map_str.replace("@","1 ")
        map_str = map_str.replace("T","1 ")
        map_io = StringIO(map_str)
        map_np = np.loadtxt(map_io)
        map_np = map_np[::reduction,::reduction]
        print(f'Map shape: {x_size} x {y_size}')
    return map_np, (x_size,y_size)

def getSliceMap3D(reduction=4):
    grid_slice = np.load("GridBasedPathPlanning/Data/BenchmarkMaps/RadarMap3D.npy") # zxy axis order
    grid_slice = np.transpose(grid_slice, (1,2,0))
    #print("Original map shape:", grid_slice.shape)
    grid_slice = grid_slice[::reduction, ::reduction, ::reduction].astype(float)
    #print("Reduced map shape:", grid_slice.shape)
    shape = grid_slice.shape
    return grid_slice, shape

def getColorMap():
    cmap = mpl.colormaps['magma'].resampled(255)
    cmap = (1-cmap(np.linspace(0, 1, 256)))*255
    cmap = np.delete(cmap,-1,axis=1).tolist()
    return cmap


def generateRandomTerrain_Perlin3D(shape:tuple|np.ndarray, scale=40, octaves=8, persistence=0.5, lacunarity=2.0, seed=0):
    '''
    ### Generate random 3D terrain from 3D Perlin noise -> Cave like structures

    ### Parameters:
    - shape:        Size of the terrain array
    - scale:        Adjust scale to control the size of features
    - octaves:      Adjust octaves to control the level of detail
    - persistence:  Adjust persistence to control the roughness of the terrain
    - lacunarity:   Adjust lacunarity to control the variation in frequencies
    - seed:         Seed for random generation (change for different terrain)
    '''
    terrain = np.zeros(shape)
    for x, y, z in itertools.product(*map(range,shape)):
        terrain[x, y, z] = pnoise3(x/scale, y/scale, z/scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=shape[0], repeaty=shape[1], repeatz=shape[2],
                                    base=seed)
    terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) # Normalize terrain data to range [0, 1]
    threshold = 0.5 # Threshold the data to create a binary representation
    terrain_data_binary = (terrain > threshold).astype(int)
    return terrain_data_binary.astype(float)


def generateRandomTerrain_Simplex2D(shape:tuple|np.ndarray, scale=10, octaves=4, persistence=0.5, lacunarity=2.0, seed=0):
    '''
    ### Generate random 3D terrain from 2D Simplex noise -> Above ground structures

    ### Parameters:
    - shape:        Size of the terrain array
    - scale:        Adjust scale to control the size of features
    - octaves:      Adjust octaves to control the level of detail
    - persistence:  Adjust persistence to control the roughness of the terrain
    - lacunarity:   Adjust lacunarity to control the variation in frequencies
    - seed:         Seed for random generation (change for different terrain)
    '''
    terrain = np.zeros(shape)
    for x, y in itertools.product(*map(range,shape[:2])):
        noise_value = snoise2(x/scale, y/scale,
                              octaves=octaves,
                              persistence=persistence,
                              lacunarity=lacunarity,
                              base=seed)
        terrain_height = int((noise_value + 1) / 2 * shape[2]) # [-1,1] -> [0,1]
        terrain[x, y, :terrain_height] = 1
        
    return terrain


def getListOfHighestMountains(terrain, n=1, proximity_divisor = 3): # Threshold distance to filter nearby peaks (in grid units)
    height_map = terrain.shape[2] - 1 - np.argmax(terrain[:,:,::-1], axis=2)
    heights_sorted = np.unique(height_map)[::-1]

    def is_too_close(new_point, existing_points, threshold):
        for point in existing_points:
            if np.linalg.norm(new_point-np.array(point[0:2]),ord=1) <= threshold:
                return True
        return False

    # Find the N highest mountain peaks with respect to proximity
    found_peaks = []
    proximity_threshold = np.max(terrain.shape)/proximity_divisor

    for height in heights_sorted:
        potential_peaks = np.argwhere(height_map == height)

        for peak in potential_peaks:
            # If no peak has been added yet or it's far enough from existing peaks
            if not is_too_close(peak, found_peaks, proximity_threshold):
                found_peaks.append((*peak,height)) # x,y,z
            
            # Stop if we have found N peaks
            if len(found_peaks) == n:
                return found_peaks
    

def saveData_Numpy(folder = 'GridBasedPathPlanning/Data/Processing/', overwrite=True, **kvargs):
    folder = Path(folder)
    os.makedirs(folder, exist_ok=True)

    for key, value in kvargs.items():
        filename = key

        if not overwrite: # Check if file already exists and increment the name if needed
            counter = 0
            while os.path.exists(folder/filename):
                filename = f"{key}_{counter}"
                counter += 1

        np.savez_compressed(folder/filename, **{key:value})

def saveData_Pickle(folder = 'GridBasedPathPlanning/Data/Processing/', overwrite=True, **kvargs):
    folder = Path(folder)
    os.makedirs(folder, exist_ok=True)

    for key, value in kvargs.items():
        filename = f"{key}.pkl"

        if not overwrite: # Check if file already exists and increment the name if needed
            counter = 0
            while os.path.exists(folder/filename):
                filename = f"{key}_{counter}.pkl"
                counter += 1

        with open(folder/filename, 'wb') as f:
            pickle.dump(value, f)

def loadData_Pickle(folder = 'GridBasedPathPlanning/Data/Processing/', file = 'plans.pkl'):
    with open(Path(folder)/file, 'rb') as f:
        return pickle.load(f)

def plotwMatplotLib_3D(grid_seq, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(grid_seq[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.plot([x[0]+0.5 for x in path], [y[1]+0.5 for y in path], [z[2]+0.5 for z path],"r-")
    #plt.savefig(f'fig/{0}.png',bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()

def plotwMatplotLib_2D(grid_seq, extracted_path): # DEPREATED
    for i in range(grid_seq.shape[0]):
        figure, axes = plt.subplots()
        axes.imshow(np.transpose(grid_seq[i]), cmap='magma', vmin=0, vmax=1, aspect='equal', origin='lower')
        if extracted_path is not None:
            if i < len(extracted_path):
                Drawing_colored_circle = plt.Circle((extracted_path[i][0], extracted_path[i][1]), 0.5)
                axes.add_artist(Drawing_colored_circle)
        #plt.savefig(f'fig/{i}.png',bbox_inches='tight',transparent=True, pad_inches=0)
        plt.show()


def GridToRGB(grid, normalized_out=False):
    grid_rgb = np.zeros((*grid.shape, 3))

    # Condition 1: Values <= 0.1 (White)
    grid_rgb[grid <= 0.1] = [1, 1, 1]  # White (normalized)

    # Condition 2: 0.1 < Values <= 0.9 (HUE-based color)
    hue_mask = (grid > 0.1) & (grid <= 0.9)
    hue_values = grid[hue_mask]
    
    # Only process hue_values if they exist
    if hue_values.size > 0:
        # Convert these hue values to RGB using colorsys.hsv_to_rgb
        rgb_values = np.array([colorsys.hsv_to_rgb(hue, 1, 1) for hue in hue_values])
        grid_rgb[hue_mask] = rgb_values

    # Condition 3: Values > 0.9 (Black)
    grid_rgb[grid > 0.9] = [0, 0, 0]  # Black

    # If output should not be normalized, scale to 0-255
    if not normalized_out:
        grid_rgb = (grid_rgb * 255).astype(np.uint8)
    
    return grid_rgb

def convertCoord_FloatToGrid(coord_float, shape_grid):
    ''' ##### This function converts fractional distance units to the grid system indexes '''
    coord_float = np.array(coord_float)  # tuple with float values ranging [0,1]
    shape_grid = np.array(shape_grid)
    
    if np.all(np.logical_and(0 <= coord_float, coord_float <= 1)): # In array bounds
        coord_grid = np.round(coord_float * (shape_grid-1)).astype(np.int_)
        coord_grid = np.clip(coord_grid,0,(shape_grid-1))
        return coord_grid
    else:
        print('Please provide a point with float values ranging: [0,1]')