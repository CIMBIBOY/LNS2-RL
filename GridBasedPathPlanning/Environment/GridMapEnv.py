import os, sys
import numpy as np
import itertools
import yaml

from GridBasedPathPlanning.Environment.GridMapEnv_Utils import *
from GridBasedPathPlanning.Environment.Object import Object


class GridMapEnv:
    '''
    After init and adding all required objects always run: 
        RenderAllObstacles() -> To render the added obstacles to the static and dynamic grid
        RenderGrid()         -> To merge the previously rendered static and dynamic grid to the main grid
    If you plan to use the generategridseq method this is not necessary
    '''
    def __init__(self, grid_size:tuple[int, ...]):
        self.dtype = np.float16
        self.grid_size = np.array(grid_size)
        self.grid_static  = np.zeros(grid_size, dtype=self.dtype)
        self.grid_dynamic = np.zeros(grid_size, dtype=self.dtype)
        self.grid = np.zeros(grid_size, dtype=self.dtype)
        self.time = 0
        self.objects = list()

    def CreateObstacle(self,typ,shape,data):
        self.objects.append(getattr(sys.modules["GridBasedPathPlanning.Environment.Object"], shape)(self, typ=typ,shape=shape,data=data))

    def RenderObsToGrid(self, obj):
        # Notations: BaseSys(Grid or Sub)_SubjectSys_(LowerLeft or UpperRight),    Window: the common segment
        # To be implemented: Dont attempt to draw an object if it is out of the grid

        ObjGrid_shape = obj.ObjGrid.shape
        # obj.ObjGrid_ll: ObjectGrid min side coordinate in the MainGrid coordinate system
        # obj.ObjGrid_ur: ObjectGrid max side coordinate in the MainGrid coordinate system

        obj.CheckVisibility(self.grid_size)
        if obj.Visible == True:
            MainGrid_Window_min = np.clip(obj.ObjGrid_min, a_min=0, a_max=self.grid_size)
            MainGrid_Window_max = np.clip(obj.ObjGrid_max, a_min=0, a_max=self.grid_size-1)
  
            ObjGrid_Window_min = np.clip(-obj.ObjGrid_min, a_min=0, a_max=ObjGrid_shape)
            ObjGrid_Window_max = np.clip(ObjGrid_shape - (obj.ObjGrid_max - (self.grid_size - 1)), a_min=0, a_max=ObjGrid_shape)
            
            #slices = tuple(slice(start, end) for start, end in zip(start_index, end_index))
            MainGrid_slice =      tuple(slice(start, end) for start, end in zip(MainGrid_Window_min, MainGrid_Window_max+1))
            ObjGridWindow_slice = tuple(slice(start, end) for start, end in zip(ObjGrid_Window_min, ObjGrid_Window_max))
            getattr(self,('grid_'+ obj.typ))[MainGrid_slice] += obj.ObjGrid[ObjGridWindow_slice]

    # EDIT needed
    def RenderAllObstacles(self, typ = None):
        for obj in self.objects:
            if (typ is None) or (obj.typ == typ):
                self.RenderObsToGrid(obj) # No typ argument or specified
                #print(obj.shape,obj.data)

    def StepDynamicObstacles(self):
        for obj in self.objects:
            if obj.typ == 'dynamic': obj.Step()
        self.time += 1

    def ResetDynamicObstacles(self):
        for obj in self.objects:
            if obj.typ == 'dynamic': obj.init()

    def ClearGrid(self, gridtype):
        setattr(self,gridtype, np.zeros(self.grid_size,dtype=self.dtype))

    def SetStaticGrid(self, grid_static, clip=False, convert_nonzero_to_inf=False):
        if convert_nonzero_to_inf:
            grid_static = np.where(grid_static != 0, np.inf, grid_static)
            
        if clip:
            self.grid_static = np.clip(grid_static,a_min=0, a_max=1)
        else:
            self.grid_static = grid_static

    def RenderGrid(self, clip=True):
        # This function takes the static and dynamic grids and render the merged grid
        if clip:
            self.grid = np.clip(self.grid_static + self.grid_dynamic, a_min=0, a_max=1)
        else:
            self.grid = (self.grid_static + self.grid_dynamic)

    def CellIsOccupied(self, point: np.array, gridtype='grid_static'):
        if np.all(np.logical_and(0 <= point, point < self.grid_size)):
            try:
                return getattr(self, gridtype)[tuple(point)] > 0.01
            except AttributeError:
                raise AttributeError(f"{gridtype} is not a valid attribute of self.")
        else:
            return True  # or you can raise ValueError('Position value out of the Grid')

    def GetRandomFreeCell(self, point:tuple|None=None, r:int|tuple=0, max_attempts:int=100, force_even = False):
        if point is not None:
            point = np.array(point)
            if isinstance(r, int): r = np.full(point.shape,r)

        for _ in range(max_attempts):
            if point is None: # Get a random point anywhere on the grid
                point_free = np.random.randint(0,self.grid_size-1,self.grid_size.shape[0])
            else: # Get a random point_free in r radius of the given point
                point_free = np.random.randint(low=point-r, high=point+r)
            if not self.CellIsOccupied(point_free) and (not force_even or all(x%2==0 for x in point_free)):
                # Free cell AND (not force_even OR all coords even)
                return tuple(point_free)

        raise RuntimeError(f"Failed to find a free cell within the grid/ specified radius after {max_attempts} attempts.")
               
    def GenerateGridSequence(self, FRAMES, reset_after = False, clip = False):

        grid_seq         = np.empty((FRAMES,)+self.grid.shape, dtype=self.dtype)   # List of rendered grids (numpy arrays)

        self.RenderAllObstacles(typ='static')

        for i in range(FRAMES):
            self.ClearGrid(gridtype='grid_dynamic')
            if i != 0: self.StepDynamicObstacles()
            self.RenderAllObstacles(typ='dynamic')
            self.RenderGrid(clip = clip)

            grid_seq[i] = self.grid

        if reset_after:
            self.time = 0
            self.ResetDynamicObstacles()
            self.ClearGrid(gridtype='grid_dynamic')
            self.RenderAllObstacles(typ='dynamic')
            self.RenderGrid()

        return grid_seq

    def addRandomDynamicObstacles(self, no_of_obs, d_range, alpha=1):
        for i in range(no_of_obs):
            p = np.random.rand(self.grid_size.shape[0])
            p = np.multiply(p,self.grid_size).astype(int)
            d = np.random.randint(d_range[0], d_range[1]) if isinstance(d_range,(list,tuple)) else d_range
            v = np.random.rand(self.grid_size.shape[0]) -0.5

            self.CreateObstacle(typ='dynamic', shape='Ellipsoid', data={'p': p,'d':d,'vel':v,'alpha': alpha})

    def addObjectsFromYAML(self,filepathname):

        with open(filepathname,'r') as file:
            scene = yaml.safe_load(file)

        for obj in scene['objects']:
            self.CreateObstacle(typ=obj['typ'],shape=obj['shape'],data=obj['data'])


def GenerateBoundedGridSequence(shape, path_extracted, grid_seq, corridor_diameter = 10, corridor_diameter_div=3, **kwargs):
    grid_corridor = GridMapEnv(grid_size=shape)

    i = 0
    while i < path_extracted.shape[0]:
        grid_corridor.CreateObstacle(typ='static',shape = 'Ellipsoid',data = {'p':np.array(path_extracted[i,1:]).astype(int),'d':corridor_diameter})
        i += int(corridor_diameter/corridor_diameter_div)
    grid_corridor.RenderAllObstacles(typ='static')
    grid_corridor.RenderGrid()

    grid_seq_new = np.copy(grid_seq)

    for i in range(grid_seq_new.shape[0]):
        grid_seq_new[i] = np.clip(grid_seq_new[i] + (1 - grid_corridor.grid), a_min=0, a_max=1)

    if 'return_grid_corridor' in kwargs: return grid_seq_new, grid_corridor.grid

    return grid_seq_new

def getDemoGridMapEnv(radar_alpha=1):
    grid_static, shape = getSliceMap3D(reduction=8)    # Radarmap 3D slices

    grid_static = np.transpose(grid_static, axes=(1,0,2))
    grid_static = np.concatenate([grid_static, grid_static[:,::-1,:]], axis=1)
    shape = grid_static.shape

    grid = GridMapEnv(grid_size=shape) # Create the Grid environment
    grid.SetStaticGrid(grid_static=grid_static, convert_nonzero_to_inf=True)

    grid.CreateObstacle(typ='dynamic', shape='Radar', data={'p':(10,64,14), 'a':10, 'n':1.5, 'alpha':radar_alpha})
    grid.CreateObstacle(typ='dynamic', shape='Radar', data={'p':(48,36,10), 'a':10, 'n':1.5, 'alpha':radar_alpha})
    grid.CreateObstacle(typ='dynamic', shape='Radar', data={'p':(48,90,10), 'a':10, 'n':1.5, 'alpha':radar_alpha})

    return grid
