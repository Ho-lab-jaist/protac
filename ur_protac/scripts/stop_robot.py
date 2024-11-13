#!/usr/bin/env python3
import numpy as np
from std_msgs.msg import Float64MultiArray
import rospy

rospy.init_node('ur5_controller')
cmd_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10)

for _ in range(10):
    msg = Float64MultiArray()
    cmd_publisher.publish(msg)