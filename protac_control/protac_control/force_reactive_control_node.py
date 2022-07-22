import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
import numpy as np

from protac_kinematics.kinematics import Kinematics
from .controllers import JointControllerBase

class ForceReactiveController(JointControllerBase):
    def __init__(self, namespace='/', timeout=5.0):
        super().__init__(namespace, timeout=timeout)

        # self.publisher_ = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.vel_publisher_ = self.create_publisher(Vector3, '/position_controller/velocity_setpoint', 10)

        self.subscription_upper_arm = self.create_subscription(
            Float64,
            '/protac_perception/filtered_contact_force',
            self.upperarm_force_callback,
            10)
        self.subscription_upper_arm  # prevent unused variable warning

        # load kinematics module for protac (protac_kinematics package)
        self.kinematics = Kinematics()

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ts = timer_period # sampling time

        # pe = np.array([-0.167,  0.04, 0.482])
        # ps = np.array([0.3, -0.3, 0.4])
        # pe = np.array([0.3, 0.3, 0.4])

        # controller paramters
        self.xe = np.deg2rad([-90., 0., -90., 0.]) # equiblirium position
        self.stiffness = 2. # 5.5
        self.damping_coeff = 1.5
        self.speed_limit = 0.5 # limits for velocity
        self.x_limit_upper = np.deg2rad([-45., 0., -90., 0.]) # reaction limit
        self.x_limit_lower = np.deg2rad([-135., 0., -90., 0.]) # reaction limit
        # pre-planned direction of distance
        self.reactive_direction = (self.x_limit_upper-self.xe)/np.linalg.norm(self.x_limit_upper-self.xe)
        # initialize reactive magnitude
        self.contact_force = 0.
        # initialize the generator of reactive-based joint commands
        self.trajectory_generation = self.jonit_trajectory_generation()

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
            x = self.get_joint_positions()
            # x = self.get_eef_positions()

            # desired velocity that pulls robot toward equiblirium postion
            xdot_d = self.stiffness*(self.xe - x)/self.damping_coeff
            xdot_d = np.clip(xdot_d, -self.speed_limit, self.speed_limit)
            # reactive velocity (constrained on pre-planned direction) that pushes robot away from obstacle
            xdot_r = np.clip(1.5*self.contact_force*self.reactive_direction, -0.8, 0.8)
            # sum of velocity vector (desired + reactive velocity)
            xdot = xdot_d + xdot_r
        
            # TODO: publish xdot
            self.setpoint_velocity.x = xdot[0]
            self.setpoint_velocity.y = xdot[1]
            self.setpoint_velocity.z = xdot[2]
            self.vel_publisher_.publish(self.setpoint_velocity)

            x_target = x + self.ts*xdot
            x_target = self.validate_reactive_path(x_target)
            self.get_logger().info('Tartget velocity: {0}\t Target position: {1}'.format(xdot, x_target))
            # calculate jont position (by inverse kinematics)
            joint=x_target
            # joint = self.kinematics.FindClosestIksolution(x_target, self.get_joint_positions())
            # yield joint
            yield xdot
                    
    def timer_callback(self):   
        joint_cmd = next(self.trajectory_generation)
        self.joint_msg.data = [joint_cmd[0], joint_cmd[1], joint_cmd[2], joint_cmd[3]]
        self.get_logger().info('Publishing command!')
        self.publisher_.publish(self.joint_msg)

    def upperarm_force_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        self.contact_force = msg.data
        

def main(args=None):
    rclpy.init(args=args)
    position_controller = ForceReactiveController()
    rclpy.spin(position_controller)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    position_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
