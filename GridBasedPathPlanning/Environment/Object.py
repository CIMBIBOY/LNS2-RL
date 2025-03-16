import numpy as np
import itertools
import abc

class Object():
    __metaclass__ = abc.ABCMeta
    def __init__(self, env, typ, shape, data):
        self.env = env      # get time information from the environment
        self.typ = typ      # 'static' or 'dynamic'
        self.shape = shape
        self.data = data    # Save for reset
        self.time_start = self.env.time
        self.init_common(data)  # process and init the input data from the dict

    @property
    def get_typ(self): return self.typ
    
    def init_common(self, data):
        self.ObjGrid = None
        self.ObjGrid_min = None
        self.ObjGrid_max = None
        self.Visible = True

        self.alpha = data['alpha'] if 'alpha' in data else 1
        self.gradient = data['gradient'] if 'gradient' in data else 0

    def CheckVisibility(self, grid_size:np.array):
        # If the corners of the ObjGrid are out of the MainGrid in AT LEAST one direction (x,y,z)
        dimension_out_of_grid = [(self.ObjGrid_max[i] < 0 or self.ObjGrid_min[i] >= grid_size[i]) for i in range(grid_size.shape[0])]
        self.Visible = not any(dimension_out_of_grid) # OR relation between directions

    @abc.abstractmethod
    def init():
        raise(NotImplementedError)
    
    @abc.abstractmethod
    def RenderObstacleToSubgrid():
        raise(NotImplementedError)
    
    @abc.abstractmethod
    def Step():
        raise(NotImplementedError)
    
    def getCurrentPosFromPath(self): # 'path' is a numpy array with rows: t,x,y,z
        if self.env.time - self.time_start <= self.data['path'][-1,0]:
            return self.data['path'][self.env.time - self.time_start,1:].astype(np.int32)
        else:
            return self.data['path'][-1,1:].astype(np.int32)
    
    
class Ellipsoid(Object):
    def __init__(self, env, typ, shape, data):
        super().__init__(env, typ, shape, data)
        self.init()
        self.RenderObstacleToSubgrid()

    def init(self):
        if 'path' in self.data:
            self.center = self.getCurrentPosFromPath()
        else:
            self.center = np.array(self.data['p'])
            self.vel = np.array(self.data['vel']) if 'vel' in self.data else np.zeros(self.center.size)            

        self.diameter = np.array(self.data['d']) if 'd' in self.data else 1
        if self.diameter.size != self.center.size: # int or float
            self.diameter = np.full(self.center.size, self.diameter)


        self.boundbox_dim = np.array([self.diameter[i] if (self.diameter[i]%2 == 1) else (self.diameter[i]+1) for i in range(self.diameter.size)])
        self.boundbox_center = (self.boundbox_dim/2).astype(int) # Round down
        self.ObjGrid_min = self.center - self.boundbox_center
        self.ObjGrid_max = self.center + self.boundbox_center

    def RenderObstacleToSubgrid(self):
        self.ObjGrid = np.zeros(self.boundbox_dim)

        boundbox_ranges = [range(i) for i in self.boundbox_dim]
        coordinates = itertools.product(*boundbox_ranges)
        for coord in coordinates:
            distance = sum([ (coord[i]-self.boundbox_center[i])**2 / (self.diameter[i]/2)**2 for i in range(self.diameter.size) ])
            if distance <= 1: self.ObjGrid[coord] = self.alpha - distance*self.gradient*2
        self.ObjGrid = self.ObjGrid.clip(min = 0)

    def Step(self):
        if "path" not in self.data:
            if not self.Visible: self.vel = -self.vel   # If an object goes ot of frame reverse its direction
            self.center = self.center + self.vel
        else:
            self.center = self.getCurrentPosFromPath()
        
        center_rounded = np.round(self.center).astype(int)
        self.ObjGrid_min = center_rounded - self.boundbox_center
        self.ObjGrid_max = center_rounded + self.boundbox_center

class Rectangle(Object):
    def __init__(self, env, typ, shape, data):
        super().__init__(env, typ, shape, data)
        self.init()
        self.RenderObstacleToSubgrid()

    def init(self):
        self.p_min = np.array(self.data['p_min'])
        self.p_max = np.array(self.data['p_max'])
        self.vel = self.data['vel'] if 'vel' in self.data else np.zeros(self.p_min.size)

        self.ObjGrid_min = np.minimum.reduce([self.p_min,self.p_max])
        self.ObjGrid_max = np.maximum.reduce([self.p_min,self.p_max])

    def RenderObstacleToSubgrid(self):
        self.ObjGrid = np.full((abs(self.p_min - self.p_max) + 1), self.alpha).clip(min = 0)

    def Step(self):
        if not self.Visible: self.vel = -self.vel   # If an object goes ot of frame reverse its direction

        self.ObjGrid_min = np.round(self.p_min + self.vel).astype(int)
        self.ObjGrid_max = np.round(self.p_max + self.vel).astype(int)

class Radar(Object):
    def __init__(self, env, typ, shape, data):
        super().__init__(env, typ, shape, data)
        self.init()
        self.RenderObstacleToSubgrid()

    def init(self):
        self.center = np.array(self.data['p'])
        self.a = self.data['a']
        self.a = np.random.randint(np.min(self.a), np.max(self.a)) if isinstance(self.a,(list,tuple)) else self.a
        self.n = self.data['n']
        self.diameter = np.full(self.center.size, 4*self.a)
        self.diameter[2] = int(self.diameter[2]/2)
        self.vel = np.array(self.data['vel']) if 'vel' in self.data else np.zeros(self.center.size)

        self.boundbox_dim = np.array([self.diameter[i] if (self.diameter[i]%2 == 1) else (self.diameter[i]+1) for i in range(self.diameter.size)])
        self.boundbox_center = (self.boundbox_dim/2).astype(int) # Round down
        self.ObjGrid_min = self.center - self.boundbox_center
        self.ObjGrid_max = self.center + self.boundbox_center

        self.level_number = 4
        self.level_frac = self.alpha/self.level_number
        self.gradient = self.alpha/2/self.a*0.7 # OVERWRITE

    def RenderObstacleToSubgrid(self):
        self.ObjGrid = np.zeros(self.boundbox_dim)

        boundbox_ranges = [range(i) for i in self.boundbox_dim]
        coordinates = itertools.product(*boundbox_ranges)
        for coord in coordinates:
            dist_xy = np.sqrt(sum([(coord[i]-self.boundbox_center[i])**2 for i in range(2)]))
            dist_xyz = np.sqrt(sum([(coord[i]-self.boundbox_center[i])**2 for i in range(3)]))
            if dist_xy <= 2*self.a:
                dist_z = np.abs(coord[2]-self.boundbox_center[2])
                z = self.a * np.sin(np.arccos((self.a-dist_xy)/self.a)) * (np.sin(np.arccos((self.a-dist_xy)/self.a)/2))**self.n
                if dist_z <= z:
                    self.ObjGrid[coord] = np.round( (self.alpha - dist_xyz*self.gradient)/self.level_frac) * self.level_frac

        self.ObjGrid = self.ObjGrid.clip(min = 0)

    def Step(self):
        if not self.Visible: self.vel = -self.vel   # If an object goes ot of frame reverse its direction

        self.center = self.center + self.vel
        center_rounded = np.round(self.center).astype(int)
        self.ObjGrid_min = center_rounded - self.boundbox_center
        self.ObjGrid_max = center_rounded + self.boundbox_center