#!/usr/bin/env python3
import pyrobotskin as rsl 
import time
import numpy as np
import rospy
from protac_perception.msg import TacLinkDeformations
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import threading
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class SkinMarkerPublisher:
    def __init__(self):
        rospy.init_node('ur_skin_marker_publisher')
        self.pub = rospy.Publisher('/taclink_markers', MarkerArray, queue_size=10)
        self.S = rsl.RobotSkin("/tmp/taclink.json")
        self.taclink_ref_frame = "wrist_3_link"
        # transformation matrix from taclink to eff
        T_taclink_eff = np.array([[0, 0, 1, 0], 
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0.036],
                                  [0, 0, 0, 1]])
        # rotation matrix from taclink to eff
        self.R_taclink_eff = T_taclink_eff[:3, :3]
        # translation vector from taclink to eff
        self.t_taclink_eff = T_taclink_eff[:3, 3]
        self.first_step = True
        self.d0 = []
        self.argmax_depth = -1
        self.deformations = []
        self.lock = threading.Lock()
        self.taclink_thread = threading.Thread(target=self.taclink_listener)
        self.taclink_thread.start()
        time.sleep(1) # wait 1 sec for the callbacks to spin
        self.publish_makers()
    
    def taclink_listener(self):
        rospy.Subscriber('/taclink/deformation', TacLinkDeformations, self.taclink_callback)
        rospy.spin()
    
    def taclink_callback(self, msg):
        self.lock.acquire()
        if self.first_step:
            self.d0 = np.array(msg.deformations.data)
            self.first_step = False
        self.deformations = np.array(msg.deformations.data)
        self.argmax_depth = msg.argmax_depth.data
        self.lock.release()

    def vec2point(self,x):
        msg = Point()
        msg.x = x[0]
        msg.y = x[1]
        msg.z = x[2]
        return msg
    
    def create_color(self, w):
        color = ColorRGBA()
        color.r = 1-w
        color.g = 1-w
        color.b = 1-w
        color.a = 1
        return color

    def transform_frame(self, x):
        return np.dot(self.R_taclink_eff, x) + self.t_taclink_eff

    def create_skin_markers(self, id):        
        # Init marker message
        mesh = Marker()
        mesh.header.frame_id = self.taclink_ref_frame
        mesh.header.stamp = rospy.Time.now()
        mesh.id = id
        mesh.action = Marker.ADD
        mesh.type = Marker.TRIANGLE_LIST
        mesh.scale.x = 1
        mesh.scale.y = 1
        mesh.scale.z = 1
        # Create grayscale color
        self.lock.acquire()
        colors = (self.deformations-self.d0)# / 100.0 
        self.lock.release()
        # Loop through faces
        faces = self.S.get_faces()
        for i in range(len(faces)):
            f_ids = faces[i]
            # Get positions
            x0 = self.transform_frame(self.S.get_taxel(f_ids[0]).get_taxel_position())
            x1 = self.transform_frame(self.S.get_taxel(f_ids[1]).get_taxel_position())
            x2 = self.transform_frame(self.S.get_taxel(f_ids[2]).get_taxel_position())
            
            # convert them to array and to a geometry point
            v0 = [x0[0], x0[1], x0[2]]
            v1 = [x1[0], x1[1], x1[2]]
            v2 = [x2[0], x2[1], x2[2]]
            p0 = self.vec2point(x0)
            p1 = self.vec2point(x1)
            p2 = self.vec2point(x2)

            # Create the colors
            c0 = self.create_color(colors[f_ids[0]])
            c1 = self.create_color(colors[f_ids[1]])
            c2 = self.create_color(colors[f_ids[2]])

            mesh.points.append(p0)
            mesh.points.append(p1)
            mesh.points.append(p2)

            mesh.colors.append(c0)
            mesh.colors.append(c1)
            mesh.colors.append(c2)

        return mesh
    
    def create_contact_centroid_marker(self, id, action):
        sphere = Marker()
        sphere.lifetime = rospy.Duration(0.5)
        sphere.header.frame_id = self.taclink_ref_frame
        sphere.header.stamp = rospy.Time.now()
        sphere.id = id
        if action==2:
            sphere.action = Marker.DELETE
            return sphere
        sphere.action = Marker.ADD
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.01
        sphere.scale.y = 0.01
        sphere.scale.z = 0.01
        self.lock.acquire()
        pos = self.transform_frame(self.S.get_taxel(self.argmax_depth).get_taxel_position())
        sphere.pose.position.x = pos[0]
        sphere.pose.position.y = pos[1]
        sphere.pose.position.z = pos[2]
        self.lock.release()
        sphere.pose.orientation.x = 0
        sphere.pose.orientation.y = 0
        sphere.pose.orientation.z = 0
        sphere.pose.orientation.w = 1
        sphere.color.a = 1
        sphere.color.g = 1

        return sphere
    
    def publish_makers(self):
        markers = MarkerArray()
        while not rospy.is_shutdown():
            markers.markers.append(self.create_skin_markers(0))
            markers.markers.append(self.create_contact_centroid_marker(1, 0))
            self.pub.publish(markers)
            time.sleep(0.1)

    
if __name__ == "__main__":
    controller = SkinMarkerPublisher()
    rospy.spin()