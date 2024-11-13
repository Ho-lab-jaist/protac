#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
from protac_perception.util.cam_control import VideoShower
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='Image Display Node')
# parser.add_argument('--show_video', action='store_true', help='Display the video stream (default: True)')
# parser.add_argument('--save_video', action='store_true', help='Save the video stream (default: True)')
# args = parser.parse_args()


class ImageSubscriber:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_subscriber', anonymous=True)
        self.lock1 = threading.Lock()
        self.lock2 = threading.Lock()
        self.lock3 = threading.Lock()
        self.lock4 = threading.Lock()
    
        self.rgb_listener_thread = threading.Thread(target=self.rgb_listener)
        self.rgb_listener_thread.start()

        self.depth_listener_thread = threading.Thread(target=self.depth_listener)
        self.depth_listener_thread.start()

        self.obstacle_listener_thread = threading.Thread(target=self.obstacle_listener)
        self.obstacle_listener_thread.start()

        self.human_listener_thread = threading.Thread(target=self.human_listener)
        self.human_listener_thread.start()

        # Retrieve a parameter named 'my_param'
        self.cam = rospy.get_param('~cam')
        self.show_video = rospy.get_param('~show_video')
        self.save_video = rospy.get_param('~save_video')

        if self.show_video:
            # Start video display thread
            self.video_dislay = VideoShower(win_name = "RGB").start()

        time.sleep(1)

        self.rgb_image = None
        self.depth_image = None
        self.obstacle_image = None

    def rgb_listener(self):
        # Subscribe to the image topics
        rospy.Subscriber('protac/rgb_image', Image, self.image_callback1)
        # Spin to keep the script running and process incoming images
        rospy.spin()

    def depth_listener(self):
        # Subscribe to the image topics
        rospy.Subscriber('protac/depth_image', Image, self.image_callback2)
        # Spin to keep the script running and process incoming images
        rospy.spin()

    def obstacle_listener(self):
        # Subscribe to the image topics
        rospy.Subscriber('protac/obstacle_image', Image, self.image_callback3)
        # Spin to keep the script running and process incoming images
        rospy.spin()

    def human_listener(self):
        # Subscribe to the image topics
        rospy.Subscriber('protac/human_image', Image, self.image_callback4)
        # Spin to keep the script running and process incoming images
        rospy.spin()
    
    def image_callback1(self, msg):
        self.lock1.acquire()
        # Create a CvBridge object to convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        # Convert the ROS Image message to an OpenCV image
        self.rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.lock1.release()

    def image_callback2(self, msg):
        self.lock2.acquire()
        # Create a CvBridge object to convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        # Convert the ROS Image message to an OpenCV image
        self.depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.lock2.release()

    def image_callback3(self, msg):
        self.lock3.acquire()
        # Create a CvBridge object to convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        # Convert the ROS Image message to an OpenCV image
        self.obstacle_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.lock3.release()

    def image_callback4(self, msg):
        self.lock4.acquire()
        # Create a CvBridge object to convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        # Convert the ROS Image message to an OpenCV image
        self.human_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.lock4.release()   

if __name__ == '__main__':

    import rospkg
    import os
    
    r = rospkg.RosPack()
    path = r.get_path('protac_perception')
    FULL_PATH = path + '/resource/output_video'

    fps = 30

    display = ImageSubscriber()
    rate = rospy.Rate(fps)

    if display.save_video:
        fname_rgb = os.path.join(FULL_PATH, '{}_output_rgb.mp4'.format(display.cam))
        fname_depth = os.path.join(FULL_PATH, '{}_output_depth.mp4'.format(display.cam))
        fname_obstacle = os.path.join(FULL_PATH, '{}_output_obstacle.mp4'.format(display.cam))
        fname_human = os.path.join(FULL_PATH, '{}_output_human.mp4'.format(display.cam))

        # Define the video writer with the codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
        out_rgb = cv2.VideoWriter(fname_rgb, fourcc, fps, (640, 480))
        out_depth = cv2.VideoWriter(fname_depth, fourcc, fps, (640, 480))
        out_obstacle = cv2.VideoWriter(fname_obstacle, fourcc, fps, (640, 480))
        out_human = cv2.VideoWriter(fname_human, fourcc, fps, (640, 480))
    
    try:
        while not rospy.is_shutdown():
            display.lock1.acquire()
            rgb_image = display.rgb_image
            display.lock1.release()

            display.lock2.acquire()
            depth_image = display.depth_image
            display.lock2.release()
            
            display.lock3.acquire()
            obstacle_image = display.obstacle_image
            display.lock3.release()

            display.lock4.acquire()
            human_image = display.human_image
            display.lock4.release()

            if (depth_image is not None 
                and obstacle_image is not None 
                and human_image is not None):
                
                if hasattr(display, "video_dislay"):               
                    image_display_top = np.concatenate((rgb_image, 
                                                        human_image), 
                                                        axis=1)
                    
                    image_display_bot = np.concatenate((depth_image, 
                                                        obstacle_image), 
                                                        axis=1)

                    image_display = np.concatenate((image_display_top,
                                                    image_display_bot), 
                                                    axis=0)

                    display.video_dislay.frame = image_display
    
                if display.save_video:
                    # Write the frame to the video file
                    out_rgb.write(rgb_image)
                    out_depth.write(depth_image)
                    out_obstacle.write(obstacle_image)
                    out_human.write(human_image)

            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Node was interrupted. Shutting down...")
    
    finally: # Clean up code if needed
        if hasattr(display, "video_dislay"):
            display.video_dislay.stop()
        
        if display.save_video:
            out_rgb.release()
            out_depth.release()
            out_obstacle.release()
            out_human.release()
        
        rospy.loginfo("Node shut down")