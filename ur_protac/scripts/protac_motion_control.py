#!/usr/bin/env python3

"""
Test QP controller for motion with contact constraint
"""
import threading
import numpy as np
import time
import cvxpy as cp
import scipy.linalg

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from protac_perception.msg import ProTacInfo
from protac_perception.srv import ProTacStateControl
from ur_protac.msg import ControllerInfo


class Ur5eController:
    def __init__(self):
        rospy.init_node('motion_control')
        self.cmd_publisher = rospy.Publisher('/joint_group_vel_controller/command', 
                                             Float64MultiArray, 
                                             queue_size=10)
        
        self.lock = threading.Lock()
        robot = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(robot, "base_link", "wrist_3_link")
        self.num_joints = 6
        self.dof = 3
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

        self.J_tcp = [] # Jacobian 
        self.J_e = []
        self.q = [] # Current joint positions
        self.x = [] # Current end-effector pose
        self.dt = 0.01 # sampling time
        
        self.depth_threshold = 7. # unit: mm
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

        self.set_qp_solver()

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
        # compute forward kinematics
        self.x = self.kdl_kin.forward(self.q)
        self.T_base_tcp = np.array(self.x)
        # Compute tcp jacobian
        self.J = self.kdl_kin.jacobian(self.q)
        self.J_tcp = np.array(self.J)
        # compute end-effector jacobian
        re_tcp = self.T_tcp_e[:3, 3] # re w.r.t tcp
        re_base = self.T_base_tcp[:3, :3] @ re_tcp # re w.r.t base
        self.J_e = self.twist_addition_matrix(re_base) @ self.J_tcp
        self.is_running = True
        self.lock.release()
    
    def protac_info_callback(self, msg):
        self.lock.acquire()
        self.contact_depth = msg.max_depth.data
        self.xc = 0.001*np.array([msg.xc.x, msg.xc.y, msg.xc.z]) # unit: m
        self.skin_state = msg.state.data
        self.risk_score = msg.risk_score.data
        # self.human_distance_avg = msg.closest_distance.data
        # self.human_detection_score = msg.human_detection_score.data
        self.lock.release()

    def twist_addition_matrix(self, p):
        r = -p
        # convert a vector to skew-symmetric matrix (ssk)
        I = np.identity(3)
        S = np.array([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
        M = np.block([[I, S],
                      [np.zeros((3, 3)), I]])
        
        return M

    def compute_contact_jacobian(self, xc):
        # xc: contact locatin w.r.t to sensor frame (s)
        # caluculate xc w.r.t base frame (b)
        xc = np.array([0, 0, xc[2], 1])
        xc_base = self.T_base_tcp @ self.T_tcp_s @ xc
        self.lock.acquire()
        # distance between contact point and TCP in base
        rc_base = xc_base[:3] - self.T_base_tcp[:3, 3]
        # contact jacobian
        Jc = self.twist_addition_matrix(rc_base) @ self.J_tcp
        self.lock.release()
        return Jc

    def compute_contact_vector(self, xc, d):
        """
        xc: contact location w.r.t to sensor frame (s)
        d: contact depth
        """
        d = d*0.001 #unit: m
        nc = xc - np.array([0, 0, xc[2]])
        nc_normalized = nc / np.linalg.norm(nc) # w.r.t sensor frame
        nc_base = self.T_base_tcp[:3, :3] @ self.T_tcp_s[:3, :3] @ nc_normalized # w.r.t base
        return d*nc_base, nc_base


    def first_order_gain(self, t, K, tau = 2):
        return K * (1 - np.exp(-t / tau)) 
    
    def sigmoid_gain(self, t, K, min_val = 0, max_val = 5, tau = 15):
        a = tau / (max_val - min_val)
        b = (max_val + min_val) / 2
        return K / (1 + np.exp(-a * (t - b)))

    def sigmoid_risk_response(self, d, 
                              K = 0.2, dmin = 4, dmax = 5):
        tau = 15  # Time constant
        a = tau / (dmax - dmin)
        b = (dmax + dmin) / 2
        return -K / (1 + np.exp(-a * (d - b)))

    def set_gain_matrix_3d(self, gain):
        self.K = np.diag([gain] * self.dof)
        
    def set_gain_matrix_6d(self, gain):
        self.K_full = np.diag([gain] * self.num_joints)

    def set_qp_solver(self):
        # Pre-define constant matrices (replace with actual values)
        gain = 0.1 # 0.3
        self.set_gain_matrix_3d(gain)
        self.set_gain_matrix_6d(gain)
        self.alpha = 0.5 #0.5

        # Define variable
        self.dot_q = cp.Variable(self.num_joints, name= "q_dot")  # Optimized variable

        # Define quadratic programming problem structure
        self.Q_square = cp.Parameter((self.num_joints, self.num_joints), PSD=True, name="Q_square")  # Symmetric matrix parameter
        self.c = cp.Parameter(self.num_joints, name= "c")
        objective = cp.Minimize(0.5 * cp.sum_squares(self.Q_square @ self.dot_q) + self.c @ self.dot_q)

        # Define the quadratic constraint parameters
        depth_thresh = self.depth_threshold*0.001 # unit: m
        epsilon = 0.5 * depth_thresh**2
        self.A_square = cp.Parameter((self.num_joints, self.num_joints), PSD=True, name="A_square")  # Symmetric matrix parameter
        self.b = cp.Parameter(self.num_joints, name= "b")
        self.d = cp.Parameter(nonneg=True, name="d")  # constant term
        quad_term = 0.5 * cp.sum_squares(self.A_square @ self.dot_q)
        constraints = [quad_term + self.b @ self.dot_q + 0.5 * self.d <= epsilon]  # List to store constraints
        
        # Define the problem
        self.problem = cp.Problem(objective, constraints)

    def qp_solver(self, x_d):
        self.lock.acquire()
        # Update J and delta_x based on changing conditions (for testing)
        J = self.J_e[:3, :]
        # transformation matrix at effector
        T_e = self.T_base_tcp @ self.T_tcp_e
        p_d = x_d[:3]        
        delta_x = p_d - T_e[:3, 3]
        dc = self.contact_depth # contact depth fb
        xc = self.xc # contact location fb
        self.lock.release()

        # Update the QP problem internal state (Jacobian and constraints)
        # Add a small regularization term (e.g., 1e-6) to the matrix
        regularization_term = 1e-6
        regularization_matrix = regularization_term * np.identity(J.shape[1])
        Q = J.T @ J + regularization_matrix
        self.Q_square.value = scipy.linalg.sqrtm(Q)  # Symmetric matrix for quadratic form
        self.c.value = -J.T @ self.K @ delta_x  # Assuming delta_x is fixed

        # Check if a certain condition is met and add a constraint accordingly
        if dc > 1.5:            
            contact_vector, nc = self.compute_contact_vector(xc, dc)
            # Jacobian at the contact location
            Jc = self.compute_contact_jacobian(xc)[:3, :]
            Jc_constrainted = np.array(nc.T @ Jc).reshape(1, -1)
            A = self.alpha*(Jc_constrainted.T @ Jc_constrainted)*self.alpha + regularization_matrix
            self.A_square.value = scipy.linalg.sqrtm(A)
            self.b.value = self.alpha*contact_vector.T @ Jc
            self.d.value = contact_vector.T @ contact_vector
        else:
            self.A_square.value = np.zeros((self.num_joints, self.num_joints))
            self.b.value = np.zeros(self.num_joints)
            self.d.value = 0

        # Solve the problem
        self.problem.solve()

        # Get optimal solution
        optimal_dot_q = self.dot_q.value

        return optimal_dot_q, delta_x


    def qp_solver_6D(self, x_d, control_axis):
        self.lock.acquire()
        # Update J and delta_x based on changing conditions (for testing)
        J = self.J_e
        T_e = self.T_base_tcp @ self.T_tcp_e # transformation matrix at effector
        p_d = x_d[:3]
        z_d = x_d[3:]
        el = p_d - T_e[:3, 3]
        er = np.cross(T_e[:3, control_axis], z_d)  
        delta_x = np.concatenate((el, er))
        dc = self.contact_depth # contact depth fb
        xc = self.xc # contact location fb
        self.lock.release()

        # Update the QP problem internal state (Jacobian and constraints)
        regularization_term = 1e-6
        regularization_matrix = regularization_term * np.identity(J.shape[1])
        Q = J.T @ J + regularization_matrix
        self.Q_square.value = scipy.linalg.sqrtm(Q)  # Symmetric matrix for quadratic form
        self.c.value = -J.T @ self.K_full @ delta_x  # Assuming delta_x is fixed

        # Check if a certain condition is met and add a constraint accordingly
        if dc > 1.5:            
            contact_vector, nc = self.compute_contact_vector(xc, dc)
            # Jacobian at the contact location
            Jc = self.compute_contact_jacobian(xc)[:3, :]
            Jc_constrainted = np.array(nc.T @ Jc).reshape(1, -1)
            A = self.alpha*(Jc_constrainted.T @ Jc_constrainted)*self.alpha + regularization_matrix
            self.A_square.value = scipy.linalg.sqrtm(A)
            self.b.value = self.alpha*contact_vector.T @ Jc
            self.d.value = contact_vector.T @ contact_vector
        else:
            self.A_square.value = np.zeros((self.num_joints, self.num_joints))
            self.b.value = np.zeros(self.num_joints)
            self.d.value = 0

        # Solve the problem
        self.problem.solve()

        # Get optimal solution
        optimal_dot_q = self.dot_q.value

        return optimal_dot_q, delta_x

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
    rate = rospy.Rate(100) # 100hz
    rospy.loginfo("Node is up. Ready to receive commands")
    x_pub = rospy.Publisher('/protac/controller_info', 
                            ControllerInfo, 
                            queue_size=10)

    rospy.wait_for_service('protac/skin_state_control')  # Wait for the service to be available
    skin_service = rospy.ServiceProxy('protac/skin_state_control', 
                                      ProTacStateControl)
    
    # set transparent skin state
    # for _ in range(2):
    #     response = skin_service(0.5, 1)
    #     time.sleep(0.2)
    
    try:
        max_gain = 0.5
        T_e = controller.T_base_tcp @ controller.T_tcp_e
        x_e = T_e[:3, 3]
        print(T_e)
        
        # x_home = np.array([0.13, 0.65, 0.48, 0, 1, 0]) # proximity obstacle
        # x_home = np.array([0.19, 0.65, 0.48, 0, 1, 0]) # proximity phantom hand
        # x_home = np.array([-0.14, 0.65, 0.48, 0, 1, 0]) # proximity human

        x_home = np.array([0., 0.60, 0.48, 0, 1, 0]) # proximity two cam
        
        z_d = np.array([0, 1, 0])        
        
        # p_d = np.array([-0.03, 0.65, 0.48])
        # p_d = np.array([0.03, 0.65, 0.48])
        # p_d = np.array([-0.3, 0.65, 0.48])

        p_d = np.array([0., 0.90, 0.48])
        
        # p_d = np.array([-0.04, 0.65, 0.48])
        x_d = np.concatenate((p_d, z_d))
        x_d_list = [x_d, x_home]
        
        for i in range(5):
            for xd in x_d_list:
                print("Target position: {}".format(xd[:3]))
                T_e = controller.T_base_tcp @ controller.T_tcp_e
                x_e = T_e[:3, 3]
                error = np.linalg.norm(xd[:3] - x_e)
                t_s = time.time()
                while error > 0.001:
                    # controller.lock.acquire()
                    # risk_score = controller.risk_score
                    # contact_depth = controller.contact_depth
                    # qd = controller.qdot
                    # controller.lock.release()     

                    t_now = time.time()
                    duration = t_now - t_s
                    gain = controller.sigmoid_gain(duration, 
                                                    K = max_gain, 
                                                    min_val = 0, 
                                                    max_val = 4, 
                                                    tau = 15)
                    
                    # risk_gain = controller.sigmoid_risk_response(risk_score, 
                    #                                             K = max_gain, 
                    #                                             dmin = 4.0, 
                    #                                             dmax = 4.5)

                    risk_gain = 0.
                    sum_gain = np.maximum(gain + risk_gain, 0)
                    controller.set_gain_matrix_6d(sum_gain)
                    send_msg(error, gain, risk_gain, sum_gain, x_pub)
                    qdot, delta_x = controller.qp_solver_6D(xd, control_axis=2)
                    error = np.linalg.norm(delta_x)
                    print("Distance error: {}".format(error))
                    
                    # if contact_depth > 1.5:
                    #     rospy.loginfo("Collision!")
                    #     qdot = np.array([0., 0, 0., 0., 0., 0.])
                    #     controller.set_joint_velocity(qdot)
                    #     response = skin_service(0, 1)
                    #     break
                    
                    controller.set_joint_velocity(qdot)
                    
                    # if risk_score > 4.0 and np.linalg.norm(qdot) < 0.001:
                    #     break
                    
                    rate.sleep()

            # T_e = controller.T_base_tcp @ controller.T_tcp_e
            # x_e = T_e[:3, 3]
            # error = np.linalg.norm(x_home[:3] - x_e)
            # t_s = time.time()
            # while error > 0.05:
            #     controller.lock.acquire()
            #     risk_score = controller.risk_score
            #     controller.lock.release()     
            #     t_now = time.time()
            #     duration = t_now - t_s
            #     gain = controller.sigmoid_gain(duration, 
            #                                     K = 0.5, 
            #                                     min_val = 0, 
            #                                     max_val = 4, 
            #                                     tau = 15)
            #     risk_gain = 0. 
            #     sum_gain = np.maximum(gain + risk_gain, 0)
            #     controller.set_gain_matrix_6d(sum_gain)
            #     send_msg(error, gain, risk_gain, sum_gain, x_pub)
            #     qdot, delta_x = controller.qp_solver_6D(x_home, control_axis=2)
            #     error = np.linalg.norm(delta_x)
            #     print("Distance error: {}".format(error))
            #     controller.set_joint_velocity(qdot)
            #     rate.sleep()
            # response = skin_service(0.5, 3)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Node was interrupted. Shutting down...")
    
    finally: # Clean up code if needed
        # reset velocity    
        for _ in range(5):
            controller.set_joint_velocity([0.0, 0.0, 0.0, 0, 0, 0]) 
            print("stop robot")
        rospy.loginfo("Node shut down")

# rostopic pub /joint_group_vel_controller/command std_msgs/Float64MultiArray "data: [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" 