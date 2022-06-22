import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import numpy as np

from protac_kinematics.kinematics import Kinematics
from .controllers import JointControllerBase

class InteractionController(JointControllerBase):
    def __init__(self, namespace='/', timeout=5.0):
        super().__init__(namespace, timeout=timeout)

        self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
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
        self.xe = np.array([-0.131, -0.01, 0.518]) # equiblirium position
        self.speed_limit = 0.5 # limits for velocity
        self.x_limit_upper = np.array([-0.131, 0.04, 0.518]) # reaction limit
        self.x_limit_lower = np.array([-0.131, -0.04, 0.518]) # reaction limit
        # pre-planned direction of distance
        self.reactive_direction = (self.x_limit_upper-self.xe)/np.linalg.norm(self.x_limit_upper-self.xe)
        # initialize action detected
        self.action  = "No"
        # initialize interaction velocity
        self.twist_iac = np.array([0., 0., 0., 0., 0., 0.])
        # initialize the generator of reactive-based joint commands
        self.trajectory_generation = self.jonit_trajectory_generation()
        # initialize press direction
        self.press_direction = np.array([0., 0., 0.])

        # message for joint position control
        self.joint_msg = Float64MultiArray()

        # message for velocity command
        self.setpoint_velocity = Vector3()

    def validate_reactive_path(self, x_t):
        """ Constraint the reactive path to a straigh line
        with two-end boundaries with spcific line length.
        """
        if np.dot(x_t - self.x_limit_upper, x_t - self.x_limit_lower) > 0:
            if np.linalg.norm(x_t - self.x_limit_upper) < np.linalg.norm(x_t - self.x_limit_lower):
                return self.x_limit_upper
            elif np.linalg.norm(x_t - self.x_limit_lower) < np.linalg.norm(x_t - self.x_limit_upper):
                return self.x_limit_lower
        else:
            return x_t

    def jonit_trajectory_generation(self):
        while True:
            # current state (position of the robot)
            # x = self.get_eef_positions()
            # Rsb = self.get_eef_orientation()
            cjoint = self.get_joint_positions() # current joint state
            # get jacobian body
            Jbody = self.kinematics.JacobianBody(cjoint)
            # intearaction velocity inferred from the ACTION detected
            if np.isclose(np.linalg.norm(self.twist_iac[:3]), 0.):
                jointdot = np.dot(np.linalg.pinv(Jbody[3:,:]), self.twist_iac[3:])
            else:
                jointdot = np.dot(np.linalg.pinv(Jbody), self.twist_iac)
        
            joint = cjoint + self.ts*jointdot
            self.get_logger().info('Tartget velocity: {0}'.format(np.dot(Jbody, jointdot)))
            # calculate jont position (by inverse kinematics)
            # joint = self.kinematics.FindClosestIksolution(x_target, self.get_joint_positions())
            yield joint
                    
    def timer_callback(self):   
        joint_cmd = next(self.trajectory_generation)
        self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
        # self.get_logger().info('Publishing command!')
        self.publisher_.publish(self.joint_msg)

    def action_callback(self, msg):
        x = 0.
        y = 0.
        z = 0.
        wx = 0.
        wy = 0.
        wz = 0.
        self.twist_iac = np.array([0., 0., 0., 0., 0., 0.])
        self.action = msg.data
        # self.get_logger().info('I heard: "%s"' % self.action)
        if self.action == "Stroke(-)":
            # self.twist_iac = np.array([-0.5, -0., 0., 0., 0., 0.])
            # self.twist_iac = np.array([0., 0., 0., 0., -0., -0.2])
            z = -0.2
        elif self.action == "Stroke(+)":
            # self.twist_iac = np.array([0.5, 0., 0., 0., 0., 0.])
            # self.twist_iac = np.array([0., 0., 0., 0., 0., 0.2])    
            z = 0.2
        elif self.action == "Press":
            # self.twist_iac = np.array([0., 0., 0., 0., 0., 0.])
            # self.twist_iac[3:] = 0.12*self.press_direction
            x = 0.35*(self.press_direction[0]/20)
            y = 0.35*(self.press_direction[1]/20)
        elif self.action == "My(+)":
            wy = 1.
        elif self.action == "My(-)":
            wy = -1.
        elif self.action == "Mx(+)":
            wx = 1.
        elif self.action == "Mx(-)":
            wx = -1.
        elif self.action == "No":
            pass

        rotation = 0.5*np.array([wx, wy, wz])
        velocity = np.array([x, y, z])
        self.twist_iac[:3] = rotation
        self.twist_iac[3:] = velocity


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
