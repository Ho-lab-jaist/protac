#!/usr/bin/env python3

"""
Plot data
"""
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

import rospy
import threading
import numpy as np
import time
import tf

from collections import deque
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

from protac_perception.msg import ProTacInfo
from protac_perception.srv import ProTacStateControl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from ur_protac.msg import ControllerInfo


class DataAcquisitionNode:
    def __init__(self):
        rospy.init_node('data_acquistion_node')
        
        self.lock = threading.Lock()
        robot = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")
        # transformation from end-effector to tcp
        self.T_tcp_e = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.31],
                                 [0, 0, 0, 1]])
        # transformation from skin to tcop
        self.T_tcp_s = np.array([[0,  1, 0, 0],
                                 [-1, 0, 0, 0],
                                 [0,  0, 1, 0.035],
                                 [0, 0, 0, 1]])
                
        self.joint_states_thread = threading.Thread(target=self.joint_states_listener)
        self.joint_states_thread.start()
        self.tactile_fb_thread = threading.Thread(target=self.protac_callback_listener)
        self.tactile_fb_thread.start()
        self.ref_thread = threading.Thread(target=self.controller_info_listener)
        self.ref_thread.start()
        time.sleep(1) # wait 1 sec for the callbacks to spin

        # Retrieve a parameter named 'my_param'
        self.cam = rospy.get_param('~cam')


        self.log_start = False

        rospy.loginfo("Waiting for /joint_states up")
        while not self.is_running:
            time.sleep(0.01)
        rospy.loginfo("/joint_states up")

        rospy.loginfo("Waiting for logging up")
        # self.log_start = True
        while not self.log_start:
            if np.linalg.norm(self.qdot) > 0.001:
                self.log_start = True
            time.sleep(0.01)
        rospy.loginfo("Start logging!")

    def protac_callback_listener(self):
        rospy.Subscriber('protac/protac_info', ProTacInfo, self.protac_info_callback)
        rospy.spin()

    def joint_states_listener(self):
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.spin()

    def controller_info_listener(self):
        self.error = 0.
        rospy.Subscriber('/protac/controller_info', ControllerInfo, self.protac_controller_info_callback)
        rospy.spin()

    def joint_states_callback(self, msg):
        self.lock.acquire()
        # Update q, J and x
        idx = np.array([2, 1, 0, 3, 4, 5], dtype=int)
        # Read current joint position
        self.q = np.array(msg.position)
        self.q = self.q[idx]
        # Read current joint velocity
        self.qdot = np.array(msg.velocity)
        self.qdot = self.qdot[idx]
        # compute forward kinematics
        self.x = self.kdl_kin.forward(self.q)
        self.T_base_tcp = np.array(self.x)
        self.T_e = self.T_base_tcp @ self.T_tcp_e
        self.p_e = self.T_e[:3, 3]
        self.R_e = self.T_e[:3, :3]
        self.rpy_e = tf.transformations.euler_from_matrix(self.R_e)

        self.is_running = True
        self.lock.release()
    
    def protac_info_callback(self, msg):
        self.lock.acquire()
        self.skin_state = msg.state.data
        self.contact_depth = msg.max_depth.data
        self.xc = 0.001*np.array([msg.xc.x, msg.xc.y, msg.xc.z]) # unit: m
        self.risk_score =  np.float64(msg.risk_score.data)     
        self.obs_location = np.array([msg.obs_location.x, msg.obs_location.y, msg.obs_location.z])
        self.human_detection_score = msg.human_detection_score.data
        self.closest_distance = msg.closest_distance.data

        # self.human_distance_avg = msg.closest_distance.data
        
        self.lock.release()

    def protac_controller_info_callback(self, msg):
        self.lock.acquire()
        self.error = np.float64(msg.error.data)
        self.gain = np.float64(msg.gain.data)
        self.risk_gain = np.float64(msg.risk_gain.data)
        self.sum_gain = np.float64(msg.sum_gain.data)
        self.lock.release()

if __name__ == "__main__":
    import pickle
    import rospkg
    import os
    import datetime

    exp_name = "exp04_two_cam_combination_brightness_x5"
    try:
        data_node = DataAcquisitionNode()
        rate = rospy.Rate(100)

        timestamps = []
        # Sensor data log
        skin_state_log = []
        contact_depth_log = []
        xc_log = []
        risk_score_log = []
        obs_location_log = []
        human_score_log = []
        closest_distance_log = []
        # Robot data log
        p_e_log = []
        rpy_e_log = []
        q_log = []
        qdot_log = []
        # Controller parameter log
        error_log = []
        gain_log = []
        risk_gain_log = []
        sum_gain_log = []

        while not rospy.is_shutdown():
            if data_node.log_start:
                data_node.lock.acquire()
                # Sensor data log
                skin_state = data_node.skin_state
                contact_depth = data_node.contact_depth
                xc = data_node.xc
                risk_score = data_node.risk_score
                obs_location = data_node.obs_location
                human_score = data_node.human_detection_score
                closest_distance = data_node.closest_distance
                # Robot data log
                p_e = data_node.p_e
                rpy_e = data_node.rpy_e
                q = data_node.q
                qdot = data_node.qdot
                # # Controller parameter log
                # error = data_node.error
                # gain = data_node.gain
                # risk_gain = data_node.risk_gain
                # sum_gain = data_node.sum_gain
                data_node.lock.release()

                timestamp = datetime.datetime.now()
                timestamps.append(timestamp)
                # Sensor data log
                skin_state_log.append(skin_state)
                contact_depth_log.append(contact_depth)
                xc_log.append(xc)
                risk_score_log.append(risk_score)
                obs_location_log.append(obs_location)
                human_score_log.append(human_score)
                closest_distance_log.append(closest_distance)
                # Robot data log
                p_e_log.append(p_e)
                rpy_e_log.append(rpy_e)
                q_log.append(q)
                qdot_log.append(qdot)
                # # Controller parameter log
                # error_log.append(error)
                # gain_log.append(gain)
                # risk_gain_log.append(risk_gain)
                # sum_gain_log.append(sum_gain)       

            rate.sleep()


    except rospy.ROSInterruptException:
        pass

    finally: # Clean up code if needed        
        # Create a dictionary to hold all your data
        data = {
            "timestamps": timestamps,
            "skin_state_log": skin_state_log,
            "contact_depth_log": contact_depth_log,
            "xc_log": xc_log,
            "risk_score_log": risk_score_log,
            "obs_location_log": obs_location_log,
            "human_score_log": human_score_log,
            "closest_distance_log": closest_distance_log,
            "p_e_log": p_e_log,
            "rpy_e_log": rpy_e_log,
            "q_log": q_log,
            "qdot_log": qdot_log,
            "error_log": error_log,
            "gain_log": gain_log,
            "risk_gain_log": risk_gain_log,
            "sum_gain_log": sum_gain_log
        }

        r = rospkg.RosPack()
        path = r.get_path('protac_perception')
        FULL_PATH = path + '/resource/log_data'
        SAVE_PATH = os.path.join(FULL_PATH, "{}_{}.pkl".format(data_node.cam, exp_name))

        # Save the data to a JSON file
        with open(SAVE_PATH, "wb") as file:
            pickle.dump(data, file)

        rospy.loginfo("Saved data!")