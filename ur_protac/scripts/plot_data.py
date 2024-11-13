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

from collections import deque
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

from protac_perception.msg import ProTacInfo
from protac_perception.srv import ProTacStateControl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class Plot:
    def __init__(self):
        # Create the figure and axis
        self.fig, (self.ax_depth, self.ax_error) = plt.subplots(1, 2, figsize=(16, 8))

        time_max = 120

        depth_threshold = 0.007
        ### Define objects for contact depth axis ###
        depth_threshold = depth_threshold*1000 # unit: mm
        x_max = time_max
        y_max = depth_threshold + 2
        self.ax_depth.plot([0, x_max], 
                           [depth_threshold, depth_threshold], 
                           'r-', 
                           linewidth = 2)
        (self.line_depth, 
         self.contact_depth_data, 
         self.timestamp) = self.setup_2dplot(
                                            ax = self.ax_depth,
                                            plot_title = 'Contact depth',
                                            x_label = 'Time [s]',
                                            y_label = 'depth [cm]',
                                            y_lim = (0, y_max),
                                            x_lim = (0, x_max))
        
        self.text_depth = self.ax_depth.text(0, 0, 
                                '', 
                                ha = 'center',
                                fontsize = 16,
                                transform=None)
        trans = self.ax_depth.transData + Affine2D().translate(480, 580)
        self.text_depth.set_transform(trans)


        ### Define line for goal error axis ###
        x_e_max = time_max
        y_e_max = 0.5
        (self.line_error, 
         self.error_data, 
         self.error_timestamp) = self.setup_2dplot(
                                            ax = self.ax_error,
                                            plot_title = 'Goal error',
                                            x_label = 'Time [s]',
                                            y_label = 'error [m]',
                                            y_lim = (0, y_e_max),
                                            x_lim = (0, x_e_max))
        self.text_error = self.ax_error.text(0, 0,
                                '', 
                                ha = 'center',
                                fontsize = 16,
                                transform=None)
        trans = self.ax_error.transData + Affine2D().translate(480, 580)
        self.text_error.set_transform(trans)

        self.time_start = []

    def update(self, contact_depth, error):   
        # contact depth plot
        self.draw(contact_depth,
                  self.ax_depth,
                  self.timestamp,
                  self.contact_depth_data,
                  self.line_depth)
        text_content = "d = {0:.2f} mm".format(contact_depth)
        self.text_depth.set_text(text_content)
        # self.ax_depth.relim()
        # self.ax_depth.autoscale_view()

        # error plot
        self.draw(error,
                  self.ax_error,
                  self.error_timestamp,
                  self.error_data,
                  self.line_error)
        text = "e = {0:.3f} m".format(error)
        self.text_error.set_text(text)
        # self.ax_error.relim()
        # self.ax_error.autoscale_view()

    def setup_2dplot(self,
                     ax,
                     plot_title,
                     x_label,
                     y_label,
                     y_lim,
                     x_lim):
        # Desired font size
        label_font_size = 14
        tick_font_size = 14
        font = FontProperties(family='Arial', size=label_font_size)
        
        ax.set_xlabel(x_label, fontproperties=font)
        ax.set_ylabel(y_label, fontproperties=font)
        # Set the font size of the tick labels on the x and y axes
        ax.tick_params(axis='x', labelsize=tick_font_size)
        ax.tick_params(axis='y', labelsize=tick_font_size)
        ax.set_title(plot_title, fontsize=label_font_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        line, = ax.plot([], [], 'b-', linewidth = 2)

        N = 1000  # Maximum number of samples to keep
        x = deque(maxlen=N)  # Create a deque with maximum length N
        y = deque(maxlen=N)

        return line, x, y

    def draw(self, data, ax, x, y, line):
        # Append the new data point(s) to the data arrays
        duration = (time.time_ns() - self.time_start) / 1e9
        x.append(duration)
        y.append(data)
        # Update the plot data
        line.set_data(x, y)
        # Adjust the plot limits if necessary
        # ax.relim()
        # ax.autoscale_view()

class PlotNode:
    def __init__(self):
        rospy.init_node('plot_node')
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
        self.ref_thread = threading.Thread(target=self.reference_listener)
        self.ref_thread.start()
        time.sleep(1) # wait 1 sec for the callbacks to spin

        self.plot_app = Plot()
        self.log_start = False
        ani = FuncAnimation(self.plot_app.fig, 
                            self.update_plot, 
                            frames=None, 
                            interval=30, 
                            cache_frame_data=False)
        plt.show()


    def update_plot(self, frame):
        self.lock.acquire()
        qdot = self.qdot
        depth = self.contact_depth
        error = self.error
        self.lock.release()
        
        if np.linalg.norm(qdot) > 0.001 and not self.log_start:
            self.log_start = True
            self.plot_app.time_start = time.time_ns()

        if self.log_start:    
            self.plot_app.update(depth, error)


    # def run(self):
    #     while not rospy.is_shutdown():
    #         self.lock.acquire()
    #         depth = self.contact_depth
    #         pe = self.p_e
    #         self.lock.release()

    #         error = np.linalg.norm(pe - self.p_goal)
    #         # self.plot_app.update(depth, error)
    #         # if np.abs(error - self.pre_error) > 0.001:
    #         #     self.plot_app.update(depth, error)
    #         self.pre_error = error


    def protac_callback_listener(self):
        rospy.Subscriber('protac/protac_info', ProTacInfo, self.protac_info_callback)
        rospy.spin()

    def joint_states_listener(self):
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.spin()

    def reference_listener(self):
        self.error = 0.
        rospy.Subscriber('/protac/error', Float64, self.protac_ref_callback)
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

        self.is_running = True
        self.lock.release()
    
    def protac_info_callback(self, msg):
        self.lock.acquire()
        self.contact_depth = msg.max_depth.data
        # self.xc = 0.001*np.array([msg.xc.x, msg.xc.y, msg.xc.z]) # unit: m
        # self.human_distance_avg = msg.closest_distance.data
        # self.human_detection_score = msg.human_detection_score.data
        # self.skin_state = msg.state.data
        self.lock.release()

    def protac_ref_callback(self, msg):
        self.lock.acquire()
        self.error = np.float64(msg.data)
        # print(self.error)
        self.lock.release()

if __name__ == "__main__":
    try:
        node = PlotNode()
        # node.run()
    except rospy.ROSInterruptException:
        pass