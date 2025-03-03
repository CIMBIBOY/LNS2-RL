
class Node_TJPS(object):
    '''
    Parameters:
    current: tuple  current coordinate
    parent_motion_idx   
    time: int       time of reaching current coordinate
    g: float        path cost
    h: float        heuristic cost
    '''
    def __init__(self, current: tuple, parent_motion_idx: int=None, t:int=0, g: float=0) -> None:
        self.current = current
        self.parent_motion_idx = parent_motion_idx  # parent_motion_idx (or motion_idx in case of self.Motions)
        self.t = t
        self.g = g
        self.h = None
        self.jp_data = None
    
    def __add__(self, node):
        if len(self.current) == 2:
            return Node_TJPS((self.x+node.x,self.y+node.y),
                    node.parent_motion_idx,
                    self.t+1,
                    self.g + node.g)
        else: 
            return Node_TJPS((self.x+node.x,self.y+node.y,self.z+node.z),
                    node.parent_motion_idx,
                    self.t+1,
                    self.g + node.g)

    def __eq__(self, node) -> bool:
        return ((self.current == node.current) and (self.t == node.t))
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        return self.g + self.h < node.g + node.h or \
              (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)
        
    def __str__(self) -> str:
        return "NODE: curr:{} par_idx:{} t:{} g:{} h:{} jp_data:{}" \
                .format(self.current, self.parent_motion_idx, self.t, self.g, self.h, self.jp_data)
    
    @property
    def x(self) -> float: return self.current[0]
    @property
    def y(self) -> float: return self.current[1]
    @property
    def z(self) -> float: return self.current[2]