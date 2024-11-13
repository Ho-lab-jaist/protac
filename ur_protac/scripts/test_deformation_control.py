#!/usr/bin/env python3

"""
Test the depth control along the desired normal vector 
of a given contact position
"""


from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from sensor_msgs.msg import JointState
import rospy
import threading
import numpy as np
from std_msgs.msg import Float64MultiArray
import time
from protac_map.msg import TactileControlInfo

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
        self.Kp_d = 0.002 #0.1
        self.Kp_l = 0.1 #0.1
        self.joint_states_thread = threading.Thread(target=self.joint_states_listener)
        self.joint_states_thread.start()
        self.tactile_control_thread = threading.Thread(target=self.tactile_reference_listener)
        self.tactile_control_thread.start()
        time.sleep(1) # wait 1 sec for the callbacks to spin
        
        self.is_running = False
        self.controller_started = False

    def joint_states_listener(self):
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.spin()

    def tactile_reference_listener(self):
        rospy.Subscriber('/taclink_map_info', TactileControlInfo, self.tactile_control_callback)
        rospy.spin()

    def joint_states_callback(self, msg):
        self.lock.acquire()
        # Update q, J and x
        idx = np.array([2, 1, 0, 3, 4, 5],dtype=int)
        self.q = np.array(msg.position)
        self.q = self.q[idx]
        self.J = self.kdl_kin.jacobian(self.q)
        self.x = self.kdl_kin.forward(self.q)
        self.is_running = True
        self.lock.release()
    
    def tactile_control_callback(self, msg):
        # stop robot if no contact is detected (it can be changed later)
        if not msg.contact.data: 
            if not self.controller_started:
                controller.set_cartesian_velocity([0.0, 0.0, -0.01, 0, 0, 0]) 
            else:
                print("Robot stop!")
                controller.set_cartesian_velocity([0.0, 0.0, -0.0, 0, 0, 0]) 
        else:
            rospy.loginfo("Getting data at time %s" % rospy.get_time())
            xc_d = np.array([msg.xc_d.x, msg.xc_d.y, msg.xc_d.z])
            nc_d = np.array([msg.nc_d.x, msg.nc_d.y, msg.nc_d.z])
            deformation = msg.deformation.data
            deformation_d = msg.deformation_d.data
            self.lock.acquire()
            # deformation error
            error_d = -self.Kp_d*(deformation_d - deformation)*nc_d
            # linear velocity error
            error_l = self.Kp_l*(xc_d - np.array(self.x[0:3, 3].transpose()))
            # error_l = np.array([0., 0., 0.])
            # sum of error
            error = error_d + error_l
            dx_d = np.zeros(6)
            dx_d[:3] = error
            print("Linear: {}".format(dx_d[:3]))
            self.lock.release()
            controller.set_cartesian_velocity(dx_d)
            self.controller_started = True
    
    def set_cartesian_velocity(self,dx_d):
        self.lock.acquire()
        if self.is_running:
            J_inv = np.linalg.pinv(self.J)
            dq_d = np.array(np.matmul(J_inv,np.array(dx_d))).squeeze()
            msg_vel = Float64MultiArray()
            msg_vel.data = list(dq_d)
            # print(dq_d)
            self.cmd_publisher.publish(msg_vel)
        self.lock.release()

    def move_to_initial_position(self, gain):
        xd = np.array([-0.00, 0.360, 0.100])
        nd = np.array([0, 1, 0])
        # Compute the error
        self.lock.acquire()
        el = np.array(self.x[0:3, 3].transpose()) - xd
        er = np.cross( nd, np.array(self.x[0:3, 2].transpose()) )  
        e = np.concatenate((el.squeeze(), er.squeeze()), axis=None)
        self.lock.release()
        print("Robot start moving to initial position!")
        # start the loop
        while np.sqrt(np.dot(e, e)) > 0.005:
            self.lock.acquire()
            el = np.array(self.x[0:3, 3].transpose()) - xd
            er = np.cross( nd, np.array(self.x[0:3, 2].transpose()) )  
            self.lock.release()
            e = np.concatenate((el.squeeze(), er.squeeze()), axis=None)
            dx_d = -gain*e
            #print(dx_d)
            print("Error: {}".format(np.sqrt(np.dot(e, e))))
            self.set_cartesian_velocity(dx_d)
            time.sleep(0.01)
        controller.set_cartesian_velocity([0.0, 0.0, 0.0, 0, 0, 0]) 

       
if __name__ == "__main__":
    
    controller = Ur5eController()
    rospy.loginfo("Node is up. Ready to receive commands")
    controller.move_to_initial_position(0.1)
    rospy.spin()

