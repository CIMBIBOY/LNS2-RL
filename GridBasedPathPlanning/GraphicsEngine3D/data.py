from model import *

import sys;
from pathlib import Path
import numpy as np
import pickle
import itertools

class Data:
    def __init__(self, app, subfolder:str = ''):
        self.app = app
        folder = 'GridBasedPathPlanning/Data/Processing/' + subfolder + '/'

        try:
            with open(folder + 'plans.pkl', 'rb') as f: 
                self.plans = pickle.load(f)
        except: 
            self.plans = None
            print('plans.pkl not found')

        '''
        for plan in self.plans:
            if 'path_interp_MinimumSnapTrajectory' in plan:
                del plan['path_interp_MinimumSnapTrajectory']
        '''
                
        # Open grid components
        grid_static = np.load(folder + 'grid_static.npz')['grid_static']
        grid_seq = np.load(folder + 'grid_seq.npz')['grid_seq']
        self.grid_shape = grid_static.shape

        # Convert grid_static to flattened data type
        indices_set = set()
        for idx in itertools.product(*map(range,grid_static.shape)):
            if grid_static[idx] > 0.01:
                if any(i == 0 or i == max_idx-1 for i, max_idx in zip(idx, grid_static.shape)):
                    indices_set.add(idx)
                # Check each neighbor in all 6 directions if within bounds
                if idx[0]>0 and grid_static[idx] != grid_static[idx[0] - 1, idx[1], idx[2]]:
                    indices_set.add(idx)
                if idx[0]<grid_static.shape[0]-1 and grid_static[idx] != grid_static[idx[0] + 1, idx[1], idx[2]]:
                    indices_set.add(idx)
                if idx[1]>0 and grid_static[idx] != grid_static[idx[0], idx[1] - 1, idx[2]]:
                    indices_set.add(idx)
                if idx[1]<grid_static.shape[1]-1 and grid_static[idx] != grid_static[idx[0], idx[1] + 1, idx[2]]:
                    indices_set.add(idx)
                if idx[2]>0 and grid_static[idx] != grid_static[idx[0], idx[1], idx[2] - 1]:
                    indices_set.add(idx)
                if idx[2]<grid_static.shape[2]-1 and grid_static[idx] != grid_static[idx[0], idx[1], idx[2] + 1]:
                    indices_set.add(idx)

        indices = np.array(list(indices_set), dtype=int)

        grid_static[np.isinf(grid_static)] = 1
        values = np.array([grid_static[i, j, k] for i, j, k in indices])
        self.grid_static_instancelist = np.column_stack((indices, np.expand_dims(values,axis=1)))
        #indices = np.where(grid_static >= 0.01)
        #self.grid_static_instancelist = np.column_stack(( np.transpose(indices), grid_static[indices]))

        # Convert grid_seq to flattened data type (sequence of arrays containing all the instance indices and values)
        indices = np.where((grid_seq >= 0.01) & (grid_seq < np.inf))                 # determine the max number of dynamic instances PER FRAME
        max_instance_per_frame = np.max(np.bincount(indices[0]))

        swap_Y = False # Swaps the addition order of the indices for the Y axis, so the direction of transparency changes

        self.grid_seq_dynamic_instancelist = np.zeros((grid_seq.shape[0], max_instance_per_frame, 4))
        for i in range(grid_seq.shape[0]):

            grid_transpose = np.transpose(grid_seq[i], axes=(2,1,0))

            if swap_Y: grid_transpose = np.flip(grid_transpose,axis=1)

            indices = np.where((grid_transpose >= 0.01) & (grid_transpose < np.inf))

            if swap_Y: indices = (indices[0], grid_seq[i].shape[1]-1-indices[1], indices[2])

            indices = tuple(indices[::-1])
            self.grid_seq_dynamic_instancelist[i,0:len(indices[0])] = \
                np.column_stack((np.transpose(indices), grid_seq[i][indices])) # shape: time x instance x (index, value)
            