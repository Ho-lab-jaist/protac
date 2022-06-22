import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Vector3

import numpy as np
import modern_robotics as mr

from protac_kinematics.kinematics import Kinematics
from .controllers import JointControllerBase

class InteractionController(JointControllerBase):
    def __init__(self, namespace='/', timeout=5.0):
        super().__init__(namespace, timeout=timeout)

        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.velocity_cmd_publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.vel_publisher_ = self.create_publisher(Vector3, '/position_controller/velocity_setpoint', 10)

        self.subscription_action = self.create_subscription(
            String,
            '/protac_perception/action',
            self.action_callback,
            10)
        self.subscription_action  # prevent unused variable warning

        self.subscription_press_direction = self.create_subscription(
            Vector3,
            '/protac_perception/press_direction',
            self.press_direction_callback,
            10)
        self.subscription_press_direction  # prevent unused variable warning

        # load kinematics module for protac (protac_kinematics package)
        self.kinematics = Kinematics()

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ts = timer_period # sampling time

        # pe = np.array([-0.167,  0.04, 0.482])
        # ps = np.array([0.3, -0.3, 0.4])
        # pe = np.array([0.3, 0.3, 0.4])

        # controller paramters
        self.xgoal = np.array([0., 0.1, 0.00, 1.]) # goal position w.r.t body frame
        self.speed_limit = 0.5 # limits for velocity
        self.K_omega = 0.8
        self.K_velocity = 0.8
        self.Rws = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Tws = mr.RpToTrans(self.Rws, np.zeros(3))
        thetag = 30
        self.Rgoal = np.dot(np.transpose(self.Rws), np.array([[0, -np.sin(np.deg2rad(thetag)), np.cos(np.deg2rad(thetag))], 
                                                              [0, np.cos(np.deg2rad(thetag)), np.sin(np.deg2rad(thetag))], 
                                                              [-1, 0, 0]]))
        self.pseudo_contact_position = np.array([0., 0., 0.0]) # reference to TacLink frame
        
        # self.Tsb_init = np.array([[0.707, -0., 0.707, 0.255], [0., 1., 0., -0.01], [-0.707, 0., 0.707, 0.814], [0., 0., 0., 1.]])

        # initialize action detected
        self.action  = "No"
        # initialize interaction velocity
        self.twist_iac = np.array([0., 0., 0., 0., 0., 0.])
        # initialize the generator of reactive-based joint commands
        self.trajectory_generation = self.jonit_command_generation()
        # initialize press direction
        self.press_direction = np.array([0., 0., 0.])

        # message for joint position control
        self.joint_msg = Float64MultiArray()

        # message for velocity command
        self.setpoint_velocity = Vector3()

    def jonit_command_generation(self):
        self.Tsb_init = mr.RpToTrans(self.get_eef_orientation(), self.get_eef_positions())
        print(np.dot(self.Tws, self.Tsb_init))
        while True:
            """
            T0DO: Procedure for 2D plannar object pushing
            1. Update the current contact position and direction
            2. Update adjoint transformation [Ad_Tcb]
            3. Update contact contact Jacobian Jc = [Ad_Tcb]*Jb
            4. Update the current angle (theta)
            5. Update angle deviation (theta_d-theta)
            6. Update the twist at the contact point [wx, 0, 0, 0, 0, 0, 0]
            """            

            # update the contact position, and direction here: x = self.get_eef_positions()
            x = self.get_eef_positions()
            # get the current end-effector orientation
            Rsb = self.get_eef_orientation()
            Tsb = mr.RpToTrans(Rsb, x)
            Tbprimeb = np.dot(mr.TransInv(Tsb), self.Tsb_init)

            # calculate the matrix of required rotation quantity
            R = np.dot(np.transpose(Rsb), self.Rgoal)
            # calculate so3matrix
            so3mat = mr.MatrixLog3(R)
            # calculate so3ToVec
            omg = mr.so3ToVec(so3mat)
            # calculate AxisAnge3 and get the required rotation angle
            [omghat, theta] = mr.AxisAng3(omg)
            print(omghat, np.rad2deg(theta))
            # update twist TODO: important please uncomment it: 
            self.twist_iac[:3] = self.K_omega*theta*omghat
            self.twist_iac[3:] = self.K_omega*theta*np.cross(np.array([0., 0., -0.761]), omghat)
            # get the current joint state
            cjoint = self.get_joint_positions() # current joint state
            # Tbc: the transformation matrix from the contact position w.r.t to body frame
            Tbc = mr.RpToTrans(np.identity(3), self.pseudo_contact_position)
            # [Ad_Tcb]: adjoint
            Ad_Tcb = mr.Adjoint(mr.TransInv(Tbc))


            # update linear velocity direction
            # Twb = np.dot(self.Tws, Tsb)
            # xgoal_W = np.array([self.xgoal[0], self.xgoal[1], self.xgoal[2], 1])
            # xgoal_B = np.dot(mr.TransInv(Twb), xgoal_W)[:3] # x goal w.r.t {B} body frame
            xcontact = self.pseudo_contact_position # should be updated in realtime
            xgoal = np.dot(Tbprimeb, self.xgoal)[:3]
            
            # normalize to the unit vector
            linear_velocity = (xgoal - xcontact) # w.r.t {C} frame
            # self.twist_iac[3:] = self.K_velocity*linear_velocity
            # self.get_logger().info('Goal position: {0}'.format(linear_velocity))
            # self.get_logger().info('Distance to goal position: {0}'.format(np.linalg.norm(linear_velocity)))
            # get jacobian body
            Jbody = self.kinematics.JacobianBody(cjoint)
            # update contact Jacobian
            Jcontact = np.dot(Ad_Tcb, Jbody)
            # intearaction velocity inferred from the ACTION detected
            jointdot = np.dot(np.linalg.pinv(Jcontact), self.twist_iac)
            # self.get_logger().info('Tartget twist: {0}'.format(np.dot(Jcontact, jointdot)))

            joint = cjoint + self.ts*jointdot
            self.get_logger().info('Tartget joint velocity: {0}'.format(np.rad2deg(jointdot)))
            # calculate jont position (by inverse kinematics)
            # joint = self.kinematics.FindClosestIksolution(x_target, self.get_joint_positions())
            yield joint
                    
    def timer_callback(self):   
        joint_cmd = next(self.trajectory_generation)
        self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
        self.publisher_.publish(self.joint_msg)

        # self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
        # self.velocity_cmd_publisher_.publish(self.joint_msg)


    def action_callback(self, msg):
        pass

    def press_direction_callback(self, msg):
        self.press_direction[0] = msg.x
        self.press_direction[1] = msg.y
        self.press_direction[2] = msg.z

def main(args=None):
    rclpy.init(args=args)
    position_controller = InteractionController()
    rclpy.spin(position_controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    position_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
