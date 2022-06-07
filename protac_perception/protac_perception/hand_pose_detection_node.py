import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import Float64
import numpy as np

from collections import deque
import time
import threading

import mediapipe as mp

class VideoShower():
    def __init__(self, frame=None, win_name="Video"):
        """
        Class to show frames in a dedicated thread.

        Args:
            frame (np.ndarray): (Initial) frame to display.
            win_name (str): Name of `cv2.imshow()` window.
        """
        self.frame = frame
        self.win_name = win_name
        self.stopped = False

    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Method called within thread to show new frames.
        """
        while not self.stopped:
            # We can actually see an ~8% increase in FPS by only calling
            # cv2.imshow when a new frame is set with an if statement. Thus,
            # set `self.frame` to None after each call to `cv2.imshow()`.
            if self.frame is not None:
                cv2.imshow(self.win_name, self.frame)
                self.frame = None

            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        cv2.destroyWindow(self.win_name)
        self.stopped = True


class HandPoseDetection(Node):

    def __init__(self):
        super().__init__('hand_pose_detection')

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
        Image, 
        '/cam3/video_frames', 
        self.listener_callback, 
        10)


    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        frame = self.br.imgmsg_to_cv2(data)

def main(args=None):
    rclpy.init(args=args)

    hand_pose_detection = HandPoseDetection()

    rclpy.spin(hand_pose_detection)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hand_pose_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()