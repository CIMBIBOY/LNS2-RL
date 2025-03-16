import numpy as np
from GridBasedPathPlanning.Plan.Node import Node_TJPS

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

motions_yxw = [ Node_TJPS((0, 1),  0, 0, 1),   # VERTICAL +
                Node_TJPS((0, -1), 1, 0, 1),   # VERTICAL -
                Node_TJPS((1, 0),  2, 0, 1),   # HORIZONTAL +
                Node_TJPS((-1, 0), 3, 0, 1),   # HORIZONTAL -
                Node_TJPS((0, 0),  4, 0, 1)]   # WAIT -> Last action in list !!!

motions_2D =  [ Node_TJPS(( 1, 0), 0, 0, 1),  # X+
                Node_TJPS((-1, 0), 1, 0, 1),  # X-
                Node_TJPS(( 0, 1), 2, 0, 1),  # Y+
                Node_TJPS(( 0,-1), 3, 0, 1),  # Y-
                Node_TJPS(( 1, 1), 4, 0, 1),  # X+Y+
                Node_TJPS(( 1,-1), 5, 0, 1),  # X+Y-
                Node_TJPS((-1, 1), 6, 0, 1),  # X-Y+
                Node_TJPS((-1,-1), 7, 0, 1),  # X-Y-
                Node_TJPS(( 0, 0), 8, 0, 1)]  # WAIT

motions_xyzw =[ Node_TJPS((1, 0, 0), 0, 0, 1), # X+
                Node_TJPS((-1,0, 0), 1, 0, 1), # X-
                Node_TJPS((0, 1, 0), 2, 0, 1), # Y+
                Node_TJPS((0,-1, 0), 3, 0, 1), # Y-
                Node_TJPS((0, 0, 1), 4, 0, 1), # Z+
                Node_TJPS((0, 0,-1), 5, 0, 1), # Z-
                Node_TJPS((0, 0, 0), 6, 0, 1)] # WAIT

motions_3D = [  Node_TJPS(( 1, 0, 0), 0, 0, 1), # X+
                Node_TJPS((-1, 0, 0), 1, 0, 1), # X-
                Node_TJPS(( 0, 1, 0), 2, 0, 1), # Y+
                Node_TJPS(( 0,-1, 0), 3, 0, 1), # Y-
                Node_TJPS(( 0, 0, 1), 4, 0, 1), # Z+
                Node_TJPS(( 0, 0,-1), 5, 0, 1), # Z-
                Node_TJPS(( 1, 1, 0), 6, 0, sqrt2), # X+Y+
                Node_TJPS(( 1,-1, 0), 7, 0, sqrt2), # X+Y-
                Node_TJPS(( 1, 0, 1), 8, 0, sqrt2), # X+Z+
                Node_TJPS(( 1, 0,-1), 9, 0, sqrt2), # X+Z-
                Node_TJPS((-1, 1, 0), 10, 0, sqrt2), # X-Y+
                Node_TJPS((-1,-1, 0), 11, 0, sqrt2), # X-Y-
                Node_TJPS((-1, 0, 1), 12, 0, sqrt2), # X-Z+
                Node_TJPS((-1, 0,-1), 13, 0, sqrt2), # X-Z-
                Node_TJPS(( 0, 1, 1), 14, 0, sqrt2), # Y+Z+
                Node_TJPS(( 0, 1,-1), 15, 0, sqrt2), # Y+Z-
                Node_TJPS(( 0,-1, 0), 16, 0, sqrt2), # Y-Z+
                Node_TJPS(( 0,-1,-1), 17, 0, sqrt2), # Y-Z-
                Node_TJPS(( 1, 1, 1), 18, 0, sqrt3), # X+Y+Z+
                Node_TJPS(( 1, 1,-1), 19, 0, sqrt3), # X+Y+Z-
                Node_TJPS(( 1,-1, 1), 20, 0, sqrt3), # X+Y-Z+
                Node_TJPS(( 1,-1,-1), 21, 0, sqrt3), # X+Y-Z-
                Node_TJPS((-1, 1, 1), 22, 0, sqrt3), # X-Y+Z+
                Node_TJPS((-1, 1,-1), 23, 0, sqrt3), # X-Y+Z-
                Node_TJPS((-1,-1, 1), 24, 0, sqrt3), # X-Y-Z+
                Node_TJPS((-1,-1,-1), 25, 0, sqrt3), # X-Y-Z-
                Node_TJPS(( 0, 0, 0), 26, 0, 1)] # WAIT