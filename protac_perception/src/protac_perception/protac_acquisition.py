#!/usr/bin/env python3
import numpy as np
import argparse
import rospy
from collections import deque
import time
import threading

from protac_perception.msg import ProTacInfo
from protac_perception.srv import ProTacStateControl
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from protac_perception.tacsense import TactileProcessor
from protac_perception.object_detection import ObjectDetection
from protac_perception.proxsense import DepthSensing

from protac_perception.util.cam_control import CameraControl, VideoShower
from protac_perception.skin_control.optical_skin_control import OpticalSkinControl


# parser = argparse.ArgumentParser(description='TacLink Acquisition Node')
# parser.add_argument('--cam_id', type=int, help='Camera bus id: Default = 0', default = 0)
# parser.add_argument('--skin_state', type=int, help='Camera bus id: Default = 1', default = 1)
# args = parser.parse_args()


class ProTacNode():
    def __init__(self):
        rospy.init_node('protac_acquisition', anonymous=True)
        self.pub = rospy.Publisher('protac/protac_info', ProTacInfo, queue_size=10)
        # Create publishers for the image topics
        self.rgb_pub = rospy.Publisher('protac/rgb_image', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('protac/depth_image', Image, queue_size=10)
        self.obstacle_pub = rospy.Publisher('protac/obstacle_image', Image, queue_size=10)
        self.human_pub = rospy.Publisher('protac/human_image', Image, queue_size=10)


        model_file = 'tacnet_rand_sim_single_binary_input_05-30-23.pt'
        self.tactile_processor = TactileProcessor(num_of_nodes = 621,
                                            model_file = model_file)
        self.init_position = self.tactile_processor.init_positions

        # Object / Human detector
        self.od_processor = ObjectDetection()

        # Depth sensing
        self.depth_processor = DepthSensing(show_video = False,
                                            output = False,
                                            raw_append = False)


        # Retrieve a parameter named 'my_param'
        cam_id = rospy.get_param('~cam_id')
        self.skin_state = rospy.get_param('~skin_state')

        # Create the video capture object
        self.cam = CameraControl(cam_id = cam_id, 
                            exposure_mode = "manual",
                            undistored = False,
                            os = 'ubuntu',
                            exposure_value = 152) #152
        LED_PORT = '/dev/ttyUSB1'
        PDLC_PORT = '/dev/ttyUSB2'
        self.optical_control = OpticalSkinControl(self.cam, 
                                            pdlc_serial_port = PDLC_PORT,  
                                            led_serial_port = LED_PORT,
                                            pdlc_type = "normal")

        # start switching transparency periodically
        # self.optical_control.start_switching(dt = 0.5)

        # start control skin state servicef
        self.service_thread = threading.Thread(target=self.skin_state_control_service)
        self.service_thread.start()

        # Start video display thread
        # self.video_dislay = VideoShower(win_name = "RGB").start()
        # self.object_detection_dislay = VideoShower(win_name="Display").start()

        self.max_contact_depths = deque(maxlen = 10)
        self.avg_max_depth = 0.
        self.contact_location = [0., 0., 0.]

        self.human_detected_signals = deque(maxlen = 50)
        self.human_detected_score = 0.

        self.human_distance_signals = deque(maxlen = 10)
        self.human_distance_avg = 0.

        self.detected_frame = None
        self.object_detected = False

        # Depth sensing
        self.risk_scores = deque(maxlen = 15)
        self.avg_risk_score = 0.
        self.closest_distances = deque(maxlen = 15)
        self.avg_closest_distance = np.inf
        self.obs_location = np.array([np.inf, np.inf, np.inf])
        self.depth_display = None
        self.obstacle_display = None


    def skin_state_control_service(self):
        service = rospy.Service('protac/skin_state_control', 
                                ProTacStateControl, 
                                self.switch_skin_state)
        service.spin() #returns when either service or node is shutdown

    def switch_skin_state(self, req):
        code = req.request_state_code
        if code == req.OPAQUE:
            self.optical_control.stop_switching()
            self.optical_control.set_opaque()
            return 1 if self.optical_control.skin_state == "opaque" else 2
        elif code == req.TRANSPARENT:
            self.optical_control.stop_switching()
            self.optical_control.set_transparent()
            return 1 if self.optical_control.skin_state == "transparent" else 2
        elif code == req.DUAL:
            self.optical_control.start_switching(dt = req.dual_speed)
            return 1 if self.optical_control.running else 2
        else:
            return 2

def point2msg(x):
    msg = Point()
    msg.x = x[0]
    msg.y = x[1]
    msg.z = x[2]
    return msg

if __name__ == "__main__":
    protac = ProTacNode()
    # Create a CvBridge object to convert between OpenCV images and ROS images
    bridge = CvBridge()

    rospy.wait_for_service('protac/skin_state_control')  # Wait for the service to be available
    skin_service = rospy.ServiceProxy('protac/skin_state_control', 
                                      ProTacStateControl)
    response = skin_service(0.5, protac.skin_state)
    time.sleep(0.2)

    while not rospy.is_shutdown():

        rgb_frame = protac.cam.read()

        if (protac.optical_control.skin_state == "opaque" 
            and time.time() - protac.optical_control.switch_time > 0.03
            ):
            protac.tactile_processor.set_input(rgb_frame)
            _, est_contact_depth = protac.tactile_processor.get_estimated_deformation()
            protac.max_contact_depths.append(np.max(est_contact_depth))
            protac.avg_max_depth = np.mean(protac.max_contact_depths)
            contact_node_idx = np.abs(est_contact_depth - protac.avg_max_depth).argmin()
            protac.contact_location = protac.init_position[contact_node_idx]
            if protac.avg_max_depth > 1.5:
                print("Contact Detection: {}".format(protac.avg_max_depth))


        elif (protac.optical_control.skin_state == "transparent" 
            and time.time() - protac.optical_control.switch_time > 0.03
        ):
            # protac.avg_max_depth = 0.
            # Depth sensing
            (risk_score,
             closest_distance, 
             obs_point, 
             obstacle_image, 
             depth_image) = protac.depth_processor.processing(rgb_frame)
            protac.depth_display = depth_image
            protac.obstacle_display = obstacle_image
            protac.obs_location = obs_point
            protac.risk_scores.append(risk_score)
            protac.avg_risk_score = np.mean(protac.risk_scores)
            protac.closest_distances.append(closest_distance)
            protac.avg_closest_distance = np.mean(protac.closest_distances)
            if protac.avg_risk_score > 4.5:
                print("Collision Warn, Risk score: {}".format(protac.avg_risk_score))

            # Human Detection 
            (protac.detected_frame, 
             protac.object_detected, 
             human_detected,
             human_distance) = protac.od_processor.processing(rgb_frame, 
                                                              protac.avg_risk_score)

            protac.human_detected_signals.append(int(human_detected))
            protac.human_detected_score = np.mean(protac.human_detected_signals)
            if not human_detected and protac.human_detected_score > 0.:
                prev_signal = protac.human_distance_signals[-1]
                protac.human_distance_signals.append(prev_signal)
            else:
                protac.human_distance_signals.append(human_distance)
            protac.human_distance_avg = np.mean(protac.human_distance_signals)

        msg = ProTacInfo()
        msg.header.stamp = rospy.Time.now() 
        msg.state.data = protac.optical_control.skin_state
        msg.max_depth.data = protac.avg_max_depth
        msg.xc = point2msg(protac.contact_location)
        msg.risk_score.data =  protac.avg_risk_score     
        msg.obs_location = point2msg(protac.obs_location)
        msg.closest_distance.data = protac.avg_closest_distance
        msg.human_detection_score.data = protac.human_detected_score
        msg.object_detected.data = protac.object_detected
        protac.pub.publish(msg)
        
        # DISPLAY VIDEO FRAMES    
        ros_image_rgb = bridge.cv2_to_imgmsg(rgb_frame, encoding="bgr8")
        protac.rgb_pub.publish(ros_image_rgb)
        if (protac.depth_display is not None 
            and protac.obstacle_display is not None and 
            protac.detected_frame is not None):
            # image_display = np.concatenate((rgb_frame, protac.depth_display, protac.obstacle_display), axis=1)
            # protac.video_dislay.frame = image_display
            # Create ROS Image messages from the OpenCV images
            ros_image_depth = bridge.cv2_to_imgmsg(protac.depth_display, encoding="bgr8")
            ros_image_obstacle = bridge.cv2_to_imgmsg(protac.obstacle_display, encoding="bgr8")
            ros_image_human = bridge.cv2_to_imgmsg(protac.detected_frame, encoding="bgr8")
            # Publish the images to their respective topics
            
            protac.depth_pub.publish(ros_image_depth)
            protac.obstacle_pub.publish(ros_image_obstacle)
            protac.human_pub.publish(ros_image_human)

    print("ROS node is about to quit. Performing cleanup...")
    protac.cam.release()
    protac.optical_control.stop_switching()
    protac.optical_control.terminate_pdlc_control()
    # protac.video_dislay.stop()