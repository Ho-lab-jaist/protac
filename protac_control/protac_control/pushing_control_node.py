import rclpy
from rclpy.node import Node
import time

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from controller_manager_msgs.srv import SwitchController

import numpy as np
import modern_robotics as mr

from protac_kinematics.kinematics import Kinematics
from protac_interfaces.msg import ContactState
from .controllers import JointControllerBase

class InteractionController(JointControllerBase):
    def __init__(self, namespace='/', timeout=5.0):
        super().__init__(namespace, timeout=timeout)

        self.cli = self.create_client(SwitchController, '/controller_manager/switch_controller')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('service: {0} available'.format('/controller_manager/switch_controller'))
        self.req = SwitchController.Request()

        while not self.switch_controller(start_controllers=["joint_state_broadcaster", "velocity_controller"], 
                                         stop_controllers=["position_controller","joint_trajectory_controller"]):
            self.get_logger().info('waiting for switching controlelrs ...')
        self.get_logger().info('The controllers switched sucessfully!')
    
        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.velocity_cmd_publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.vel_publisher_ = self.create_publisher(Vector3, '/position_controller/velocity_setpoint', 10)
        self.error_publisher_ = self.create_publisher(Float64, '/protac_logger/dynamic_error', 10)
        self.contact_loc_publisher_ = self.create_publisher(Vector3, '/protac_logger/contact_location', 10)

        # # message for joint velocity control
        self.joint_msg = Float64MultiArray()
        self.joint_msg.data = [0., 0., 0., 0.]
        # for _ in range(100):
        #     time.sleep(0.01)
        #     self.velocity_cmd_publisher_.publish(self.joint_msg)

        self.subscription_action = self.create_subscription(
            String,
            '/protac_perception/action',
            self.action_callback,
            10)
        self.subscription_action  # prevent unused variable warning

        self.subscription_contact_state = self.create_subscription(
            ContactState,
            '/protac_perception/contact_state',
            self.contact_state_callback,
            10)
        self.subscription_contact_state  # prevent unused variable warning

        # load kinematics module for protac (protac_kinematics package)
        self.kinematics = Kinematics()

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ts = timer_period # sampling time

        # pe = np.array([-0.167,  0.04, 0.482])
        # ps = np.array([0.3, -0.3, 0.4])
        # pe = np.array([0.3, 0.3, 0.4])

        # controller paramters
        self.speed_limit = 0.5 # limits for velocity
        self.K_omega = 0.35 # [0.05, 0.15, 0.25, 0.35, 0.45]
        # Kw = 0.05, observe an amont of small osciallation, but can not be dimished
        # Kw = 0.15, 0.25, observe an amount of small oscillation, but can be diminished
        # Kw = 0.35, 0.45 only one strong oscilation is observed, but the condition was not improved
        self.K_velocity = 0.12 
        # 0.13, 0.14, 0.15, 0.25 oscillation observed (Kw = 0.05)
        # 0.1, 0.12 OK!, but slow (Kw = 0.05, 0.15, 0.25, 0.35)

        self.z_eff_taclink = 0.288 # z-coordinate of eff to taclink frame
        self.Rws = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        self.Tws = mr.RpToTrans(self.Rws, np.zeros(3))

        self.Rsb_init = self.get_eef_orientation()
        self.Tsb_init = mr.RpToTrans(self.Rsb_init, self.get_eef_positions())
        self.Twb_init = np.dot(self.Tws, self.Tsb_init)
        # self.get_logger().info('Twb_init: {0}'.format(np.dot(self.Tws, self.Tsb_init)))
        # self.get_logger().info('Rsb_init: {0}'.format(self.Rsb_init))

        self.target_theta = 30.
        # the target rotation matrix w.r.t the space {s} frame
        self.Rgoal = np.dot(np.transpose(self.Rws), np.array([[-np.sin(np.deg2rad(self.target_theta)), 0, np.cos(np.deg2rad(self.target_theta))], 
                                                              [np.cos(np.deg2rad(self.target_theta)), 0, np.sin(np.deg2rad(self.target_theta))], 
                                                              [0, 1, 0]]))
        self.xgoal = np.array([0.370, 0.0, -0.138, 1.]) # goal position w.r.t body frame
        # self.xgoal = np.array([0.27, 0.0, -0.3, 1.]) # goal position w.r.t body frame
        # self.xgoal = np.array([0.375, 0.0, -0.138, 1.]) # goal position w.r.t body frame
        # self.xgoal = np.array([0.3, 0.0, 0.15, 1.]) # goal position w.r.t body frame
        # self.xgoal = np.array([0.35, -0.2, -0.01022, 1.]) # goal position w.r.t {W} frame
        # # conver to to {b} body frame
        # self.xgoal = np.dot(mr.TransInv(self.Twb_init), self.xgoal)
        # self.get_logger().info('b_xgoal: {0}'.format(self.xgoal))
        # self.pseudo_contact_position = np.array([0., 0., -0.0781]) # reference to TacLink frame
        self.pseudo_contact_position = np.array([0., 0., -0.11]) # reference to TacLink frame
        # self.pseudo_contact_position = np.array([0., 0., -0.19]) # reference to TacLink frame
        
        # self.Tsb_init = np.array([[0.707, -0., 0.707, 0.255], [0., 1., 0., -0.01], [-0.707, 0., 0.707, 0.814], [0., 0., 0., 1.]])
        self.get_logger().info('x goal in space frame: {0}'.format(np.dot(self.Tsb_init, self.xgoal)))

        # initialize action detected
        self.action  = "No"
        # initialize interaction velocity
        self.twist_iac = np.array([0., 0., 0., 0., 0., 0.])
        # initialize the generator of reactive-based joint commands
        self.trajectory_generation = self.jonit_command_generation()
        # initialize press direction
        self.contact_position = np.array([0., 0., 0.178])
        # self.contact_position = np.array([0., 0., 0.098])
        self.pre_contact_position = self.contact_position
        self.num_of_contacts = 0

        # message for velocity command
        self.setpoint_velocity = Vector3()
        self.xcontact_s_msg = Vector3()
        self.dynamic_error_msg = Float64() 

    def jonit_command_generation(self):
        # self.Rsb_init = self.get_eef_orientation()
        # self.Tsb_init = mr.RpToTrans(self.Rsb_init, self.get_eef_positions())
        # self.get_logger().info('Twb_init: {0}'.format(np.dot(self.Tws, self.Tsb_init)))
        # # self.get_logger().info('xgoal_w: {0}'.format(np.dot(np.dot(self.Tws, self.Tsb_init), self.xgoal)))
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
            self.pseudo_contact_position[2] = self.contact_position[2] - self.z_eff_taclink
            
            # get current end-effector position
            x = self.get_eef_positions()
            # get the current end-effector orientation
            Rsb = self.get_eef_orientation()
            Tsb = mr.RpToTrans(Rsb, x)
            Tbprimeb = np.dot(mr.TransInv(Tsb), self.Tsb_init)
            xcontact = self.pseudo_contact_position # should be updated in realtime
            # self.get_logger().info('contact position height: {0}'.format(xcontact))
            xgoal = np.dot(Tbprimeb, self.xgoal)[:3]
            # self.get_logger().info('Updated x goal: {0}'.format(xgoal))

            self.dynamic_error_msg.data = np.linalg.norm(xgoal-xcontact)
            self.error_publisher_.publish(self.dynamic_error_msg)
            xcontact_s = np.dot(Tsb, np.append(xcontact, 1))
            self.xcontact_s_msg.x = xcontact_s[0]
            self.xcontact_s_msg.y = xcontact_s[1]
            self.xcontact_s_msg.z = xcontact_s[2]
            self.contact_loc_publisher_.publish(self.xcontact_s_msg)

            #TODO: updatade the target rotation matrix
            if np.linalg.norm((xgoal - xcontact)) != 0.:
                # normalized directional push vector in the body {b} frame
                n_push_b = (xgoal - xcontact)/np.linalg.norm((xgoal - xcontact))
                # transform it to the space {s} from the body {b} frame
                n_push_s = np.dot(self.Rsb_init, n_push_b)
                # transform this vector to another frame complementing frame for convinience
                n_push_sprime = np.dot(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), n_push_s)
                self.target_theta = np.rad2deg(np.arctan2(n_push_sprime[2], n_push_sprime[1]))
                if self.target_theta > 90.:
                    self.target_theta = self.target_theta - 180.
                # the target rotation matrix w.r.t the space {s} frame
                self.Rgoal = np.dot(np.transpose(self.Rws), np.array([[-np.sin(np.deg2rad(self.target_theta)), 0, np.cos(np.deg2rad(self.target_theta))], 
                                                                    [np.cos(np.deg2rad(self.target_theta)), 0, np.sin(np.deg2rad(self.target_theta))], 
                                                                    [0, 1, 0]]))

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
            self.get_logger().info('Omega command: {0}'.format(np.rad2deg(theta)))

            # get the current joint state
            cjoint = self.get_joint_positions() # current joint state
            # Tbc: the transformation matrix from the contact position w.r.t to body frame
            Tbc = mr.RpToTrans(np.identity(3), xcontact)
            # [Ad_Tcb]: adjoint
            Ad_Tcb = mr.Adjoint(mr.TransInv(Tbc))

            # normalize to the unit vector
            linear_velocity = (xgoal - xcontact) # w.r.t {C} frame
            self.twist_iac[3:] = self.K_velocity*linear_velocity + self.K_omega*theta*np.cross(self.pseudo_contact_position, omghat)
            # get jacobian body
            Jbody = self.kinematics.JacobianBody(cjoint)
            # update contact Jacobian
            Jcontact = np.dot(Ad_Tcb, Jbody)
            # intearaction velocity inferred from the ACTION detected
            jointdot = np.dot(np.linalg.pinv(Jcontact), self.twist_iac)

            yield jointdot
                    
    def timer_callback(self):   
        # joint_cmd = next(self.trajectory_generation)
        # self.get_logger().info('Num of contacts: {0}'.format(self.num_of_contacts))
        # if self.num_of_contacts >= 2: # halt the robot motion
        #     self.joint_msg.data = [0., 0., 0., 0.]
        # else: # run as the motion planned
        #     self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]        
        # self.velocity_cmd_publisher_.publish(self.joint_msg)
        # self.get_logger().info('Tartget joint velocity: {0}'.format(np.rad2deg(self.joint_msg.data)))

        joint_cmd = next(self.trajectory_generation)
        self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]        
        self.velocity_cmd_publisher_.publish(self.joint_msg)
        self.get_logger().info('Tartget joint velocity: {0}'.format(np.rad2deg(self.joint_msg.data)))


    def action_callback(self, msg):
        pass

    def contact_state_callback(self, msg):
        pass
        # self.num_of_contacts = msg.num_of_contacts
        # if self.num_of_contacts == 1:
        #     if np.abs(msg.contact_location[0].z - self.pre_contact_position[2]) <= 0.02:
        #         self.contact_position[0] = msg.contact_location[0].x
        #         self.contact_position[1] = msg.contact_location[0].y
        #         self.contact_position[2] = msg.contact_location[0].z
            
        #         self.pre_contact_position = self.contact_position
        
    def switch_controller(self, start_controllers=["joint_trajectory_controller"], 
                                stop_controllers=["velocity_controller"],
                                strictness=1):
        self.req.start_controllers = start_controllers
        self.req.stop_controllers = stop_controllers
        self.req.strictness = strictness
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    position_controller = InteractionController()
    try:
        rclpy.spin(position_controller)
    except Exception as e:
        raise e
    finally:
        response = position_controller.switch_controller(start_controllers=["joint_trajectory_controller"], 
                                                         stop_controllers=["velocity_controller"])
        position_controller.get_logger().info('The controllers switched {0}'.format(response.ok))
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        position_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
