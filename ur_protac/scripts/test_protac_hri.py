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
from ur_protac.msg import ControllerInfo


class Ur5eController:
    def __init__(self):
        rospy.init_node('hri_control')
        self.cmd_publisher = rospy.Publisher('/joint_group_vel_controller/command', 
                                             Float64MultiArray, 
                                             queue_size=10)
        self.lock = threading.Lock()
        robot = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")
        self.J = [] # Jacobian 
        self.q = [] # Current joint positions
        self.x = [] # Current end-effector pose
        self.dt = 0.01 # sampling time

        self.contact_signal = False
        self.deformation = 0.

        self.is_running = False
        self.controller_started = False
        self.state_switched = False

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
        self.xc = np.array([msg.xc.x, msg.xc.y, msg.xc.z])
        self.human_distance_avg = msg.closest_distance.data
        self.human_detection_score = msg.human_detection_score.data
        self.risk_score =  msg.risk_score.data     
        self.skin_state = msg.state.data
        self.lock.release()

    def sigmoid_risk_response(self, d, 
                              K = 0.2, dmin = 4, dmax = 5):
        tau = 15  # Time constant
        a = tau / (dmax - dmin)
        b = (dmax + dmin) / 2
        return -K / (1 + np.exp(-a * (d - b)))

    def sigmoid_gain(self, t, K, min_val = 0, max_val = 5, tau = 15):
        a = tau / (max_val - min_val)
        b = (max_val + min_val) / 2
        return K / (1 + np.exp(-a * (t - b)))

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
            # print("Commanded joint velocities: {0}".format(np.rad2deg(dq_d)))
            self.cmd_publisher.publish(msg_vel)


if __name__ == "__main__":
    def send_msg(error, gain, risk_gain, sum_gain, pub):
        msg = ControllerInfo()
        msg.error.data = error
        msg.gain.data = gain
        msg.risk_gain.data = risk_gain
        msg.sum_gain.data = sum_gain
        pub.publish(msg)


    controller = Ur5eController()
    rospy.loginfo("Node is up. Ready to receive commands")

    x_pub = rospy.Publisher('/protac/controller_info', 
                            ControllerInfo, 
                            queue_size=10)

    rospy.wait_for_service('protac/skin_state_control')  # Wait for the service to be available
    skin_service = rospy.ServiceProxy('protac/skin_state_control', 
                                      ProTacStateControl)
    rate = rospy.Rate(100) # 100hz
    
    # set transparent skin state
    for _ in range(2):
        response = skin_service(0.25, 2)
        time.sleep(0.3)

    normal_speed = 0.2
    t_s = time.time()
    try:
        while not rospy.is_shutdown():
            controller.lock.acquire()
            human_score = controller.human_detection_score
            risk_score = controller.risk_score
            skin_state = controller.skin_state
            controller.lock.release()
            
            t_now = time.time()
            duration = t_now - t_s
            gain = controller.sigmoid_gain(duration, 
                                           normal_speed, 
                                           min_val = 0, 
                                           max_val = 8, 
                                           tau = 15)

            # safety countermeasure
            if (human_score > 0. and risk_score > 4.5):
                print("Slow speed!")
                # risk_gain = controller.sigmoid_risk_response(risk_score, 
                #                                             K = 0.2, 
                #                                             dmin = 4.3, 
                #                                             dmax = 4.8)
                
                risk_gain = -0.2

                if not controller.state_switched:
                    response = skin_service(0.4, 3)
                    controller.state_switched = True
            else:
                if not skin_state == "transparent":
                    response = skin_service(0., 2)
                controller.state_switched = False
                risk_gain = 0.

            dq1_d = np.maximum(gain + risk_gain, 0)
            controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0]) 

            send_msg(0, gain, risk_gain, dq1_d, x_pub)

            # switch to interaction mode
            if (controller.skin_state == "opaque" 
                and controller.contact_depth > 1.5):
                response = skin_service(0., 1)
                print("Touched the skin!")
                print("Switched to tactile interaction mode!")
                while controller.skin_state == "opaque":
                    depth = controller.contact_depth
                    if depth > 1.5:
                        dq1_d = -depth/30 if controller.xc[1] > 0 else depth/30
                    else:
                        dq1_d = 0.
                    controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0])
                    send_msg(0, dq1_d, 0, 0, x_pub)
                    # print("Depth: {}".format(np.rad2deg(dq1_d))) 

            if np.rad2deg(controller.q[0]) >= 100.:
                print("Joint limit exceeded!")
                dq1_d = -normal_speed
                while np.rad2deg(controller.q[0]) > 20.:
                    controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0]) 
                    # print(np.rad2deg(controller.q[0]))
                dq1_d = 0
                controller.set_joint_velocity([dq1_d, 0.0, 0.0, 0, 0, 0])
                break
            
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Node was interrupted. Shutting down...")
    finally: # Clean up code if needed
        # reset velocity    
        for _ in range(5):
            controller.set_joint_velocity([0.0, 0.0, 0.0, 0, 0, 0]) 
            print("stop robot")
        rospy.loginfo("Node shut down")

# rostopic pub /joint_group_vel_controller/command std_msgs/Float64MultiArray "data: [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]" 