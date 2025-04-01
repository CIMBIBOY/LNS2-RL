#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion

from your_env_script import CL_MAPFEnv  # Your custom environment

def numpy_to_occupancy_grid(np_map, resolution=1.0):
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"

    msg.info.resolution = resolution
    msg.info.width = np_map.shape[1]
    msg.info.height = np_map.shape[0]
    msg.info.origin = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))

    data = []
    for row in np_map:
        for val in row:
            if val == -1:
                data.append(100)  # obstacle
            elif val == 0:
                data.append(0)    # free
            else:
                data.append(50)   # dynamic obstacle or agent
    msg.data = data
    return msg

if __name__ == '__main__':
    rospy.init_node('dynamic_map_publisher')
    pub = rospy.Publisher('/static_map', OccupancyGrid, queue_size=1)

    env = CL_MAPFEnv(env_id=0)
    env.global_set_world(cl_num_task=0)

    rate = rospy.Rate(2)  # publish at 2 Hz

    while not rospy.is_shutdown():
        env.update_dyn_map(step=1)  # updates the environment's internal map
        dynamic_map = env.merge_dynamic_map()  # get the map with dynamic obstacles
        msg = numpy_to_occupancy_grid(dynamic_map)
        pub.publish(msg)
        rate.sleep()