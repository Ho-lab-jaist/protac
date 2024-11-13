#!/usr/bin/env python3

"""
Test contact-based reactive control
"""
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from sensor_msgs.msg import JointState
import rospy
import threading
import numpy as np
from std_msgs.msg import Float64MultiArray
import time

from protac_perception.msg import ProTacInfo
from protac_perception.srv import ProTacStateControl


class Ur5eController:
    def __init__(self):
        rospy.init_node('ur5_controller')
        self.cmd_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=10)
        self.lock = threading.Lock()
        robot = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")
        self.J = [] # Jacobian 
        self.q = [] # Current joint positions
        self.x = [] # Current end-effector pose
        self.dt = 0.01 # sampling time

        # 0.1 | 0.2 | 0.3 | 0.4
        self.dq1_init = -0.30 # initial velocity

        self.prev_dq1_d = self.dq1_init # previous desired vel.
        self.prev_ddq1_d = 0. # previous desired acc.

        self.k = 0.05429059829059829
        self.M = 0.3 # mass of addmittance control
        self.B = 5 # viscous of addmittance control
        # self.k = 0.2 # stiffness of soft skin (unit: N/mm)
        # self.M = 1.5 # mass of addmittance control
        # self.B = 25 # viscous of addmittance control
        self.K = 0. # stiffness of addmittance control
        self.L = 0.50 # the length of moment arm


        self.contact_signal = False
        self.deformation = 0.

        self.is_running = False
        self.controller_started = False

        self.joint_states_thread = threading.Thread(target=self.joint_states_listener)
        self.joint_states_thread.start()
        self.tactile_control_thread = threading.Thread(target=self.tactile_reference_listener)
        self.tactile_control_thread.start()
        time.sleep(1) # wait 1 sec for the callbacks to spin

    def joint_states_listener(self):
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.spin()

    def tactile_reference_listener(self):
        rospy.Subscriber('protac/protac_info', ProTacInfo, self.protac_info_callback)
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
        # Compute jacobian
        self.J = self.kdl_kin.jacobian(self.q)
        # compute forward kinematics
        self.x = self.kdl_kin.forward(self.q)
        self.is_running = True
        self.lock.release()
    
    def protac_info_callback(self, msg):
        self.lock.acquire()
        self.contact_depth = msg.max_depth.data
        self.skin_state = msg.state.data
        self.lock.release()

    def set_joint_velocity(self, dq_d):
        """
        Set joint velocity
        Parameters:
            - dq_d: the commanded joint velocities, ndarray (6, 1)
        """
        if self.is_running:
            dq_d = np.array(dq_d)
            msg_vel = Float64MultiArray()
            msg_vel.data = list(dq_d)
            #LOG: commanded joint velocities (deg/s)
            print("Commanded joint velocities: {0}".format(np.rad2deg(dq_d)))
            self.cmd_publisher.publish(msg_vel)


if __name__ == "__main__":
    # import os
    # import datetime
    # import rospkg

    controller = Ur5eController()
    rospy.loginfo("Node is up. Ready to receive commands")
    rospy.wait_for_service('protac/skin_state_control')  # Wait for the service to be available
    skin_service = rospy.ServiceProxy('protac/skin_state_control', 
                                      ProTacStateControl)
    # start = time.time()
    # end = time.time()
    # max_time = 6
    rate = rospy.Rate(100) # 100hz

    for _ in range(2):
        dq1_d = -0.05
        controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0]) 
        time.sleep(0.2) # wait 1 sec for the callbacks to spin

    timestamps = []
    depth_logs = []
    force_logs = []
    qdot_feeback_logs = []
    qdot_command_logs = []
    while not rospy.is_shutdown():
        print(controller.contact_depth)
        # safety countermeasure
        if (controller.skin_state == "opaque" 
            and controller.contact_depth > 1.5):
            dq1_d = 0.
            controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0]) 
            response = skin_service(0, 1)
            print("Robot stop!")

        rate.sleep()
        # timestamp = datetime.datetime.now()
        # # Append the row to the data array
        # timestamps.append(timestamp)
        # depth_logs.append(controller.deformation)
        # force_logs.append(fc/controller.L)
        # qdot_command_logs.append(np.rad2deg(dq1_d))
        # qdot_feeback_logs.append(np.rad2deg(controller.qdot[0]))
        # rate.sleep()
        # end = time.time()
    
    # reset velocity    
    for _ in range(5):
        controller.set_joint_velocity([0.0, 0.0, 0.0, 0, 0, 0]) 

    # # display maximum contact depth over the contact
    # depth_logs = np.array(depth_logs)
    # print("Max. contact depth: {}".format(np.max(depth_logs)))
    # force_logs = np.array(force_logs)
    # print("Max. mesured contact force: {}".format(np.max(force_logs)))
    # timestamps_array = np.array(timestamps, dtype=object)
    # log_data_array = np.column_stack((timestamps_array, 
    #                                   depth_logs, 
    #                                   force_logs, 
    #                                   qdot_command_logs, 
    #                                   qdot_feeback_logs))
    

    # r = rospkg.RosPack()
    # path = r.get_path('ur_protac')
    # full_path = path + '/resource'

    # EXP_NAME = 'v{0}_B{1}_M{2}'.format(np.abs(controller.dq1_init),
    #                                    controller.B,
    #                                    controller.M)
    # SAVE_DIR = path
    # np.savetxt(os.path.join(SAVE_DIR, 'log_data_colli_exp_{}.csv'.format(EXP_NAME)), 
    #            log_data_array, 
    #            delimiter=',', 
    #            fmt='%s')