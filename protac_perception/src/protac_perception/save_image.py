#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
import argparse

parser = argparse.ArgumentParser(description='Image Display Node')
parser.add_argument('--depth', type=int, help='contact depth')
parser.add_argument('--position', type=int, help='contact position')
parser.add_argument('--angle', type=int, help='contact angle')
args = parser.parse_args()

class ImageSubscriber:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_subscriber', anonymous=True)
        self.lock1 = threading.Lock()

        self.rgb_listener_thread = threading.Thread(target=self.rgb_listener)
        self.rgb_listener_thread.daemon = True
        self.rgb_listener_thread.start()

        time.sleep(1)

        self.rgb_image = None

        while self.rgb_image is None:
            time.sleep(0.01)
        print("start saving data!")

    def rgb_listener(self):
        # Subscribe to the image topics
        rospy.Subscriber('protac/rgb_image', Image, self.image_callback1)
        # Spin to keep the script running and process incoming images
        rospy.spin()

    
    def image_callback1(self, msg):
        self.lock1.acquire()
        # Create a CvBridge object to convert ROS Image messages to OpenCV images
        bridge = CvBridge()
        # Convert the ROS Image message to an OpenCV image
        self.rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.lock1.release()

if __name__ == '__main__':

    import rospkg
    import os

    r = rospkg.RosPack()
    path = r.get_path('protac_perception')
    FULL_PATH = path + '/resource/output_video'

    fps = 30    
    display = ImageSubscriber()
    rate = rospy.Rate(fps)
    
    position = args.position 
    angle = args.angle
    depth = args.depth
    num_of_save_image = 5
    try:
        for i in range(num_of_save_image):
            display.lock1.acquire()
            rgb_image = display.rgb_image
            display.lock1.release()
            # angle_pos_depth_id
            fname = "{}_{}_{}_{}.jpg".format(angle, position, depth, i+1)
            fpath = os.path.join(FULL_PATH, fname)
            print(fname, rgb_image.shape)
            cv2.imwrite(fpath, rgb_image)
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Node was interrupted. Shutting down...")
    finally: # Clean up code if needed
        rospy.loginfo("Node shut down")
        rospy.loginfo("Saved images!")