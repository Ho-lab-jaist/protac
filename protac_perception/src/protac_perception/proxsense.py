import cv2
from collections import deque
import time
import numpy as np
import os

from .midas.depth_inference import DepthProcessing
import rospkg

from protac_perception.util.cam_control import VideoShower

r = rospkg.RosPack()
path = r.get_path('protac_perception')
FULL_PATH = path + '/resource/midas'


MASK_IMAGE_PATH = os.path.join(FULL_PATH, 'protac_mask_image.jpg')
mask_image = cv2.imread(MASK_IMAGE_PATH)
mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

class DepthSensing():
  def __init__(self, 
               show_video = False,
               output = False,
               raw_append = False):

    # Define ROS parameters
    self.init_parameters(show_video, 
                         output, 
                         raw_append)

    if self.show_video:
        # Start video display thread
        self.video_shower = VideoShower(win_name = "Depth").start()

    # Instantiate depth processing class
    self.depth_processing = DepthProcessing(self.model_weights, 
                                            self.model_type, 
                                            self.optimize)
    self.mask_image = mask_image
    # Number of frames to average for computing FPS.
    self.num_fps_frames = 30
    self.previous_fps = deque(maxlen=self.num_fps_frames)
    # self.depth_image = None
    # self.risk_score_avg = 0
    # self.closest_distance = 200
  
  def init_parameters(self, show_video, output, raw_append):
    # model parameters
    self.model_weights = os.path.join(FULL_PATH, "weights")
    self.model_type = "midas_v21" # dpt_large | midas_v21 | midas_v21_small
    self.optimize = True
    
    # video processing parameters
    self.show_fps = True
    self.raw_append = raw_append
    self.show_video = show_video
    self.output = output
    self.frames = list() if self.output else None

    # parameters for to-skin-distance based repulsive vector
    self.Vmax = 0.9 # maximum admissible magnitude # 0.6
    self.alpha = 6 # shape factor
    self.dmax = 0.6 # the maximum to-skin-distance at which the repulsive vector approaches zero.

  def processing(self, frame):
    loop_start_time = time.time()
    fps = int(sum(self.previous_fps) / self.num_fps_frames) if self.show_fps else None

    # Infer raw disparity map (relative distance)
    disparity_map = self.depth_processing.run(frame)

    depth_map = self.depth_processing.disparity2depth(disparity_map, dsp_const=240.)
    point_clound = self.depth_processing.back_projection(depth_map)
    point_radii = np.sqrt(point_clound[0]**2+point_clound[1]**2) 

    depth_image = self.depth_processing.display_depth(bits=1, 
                                                      fps=fps, 
                                                      raw_append=self.raw_append)

    if self.raw_append: 
        gray_depth_image = cv2.cvtColor(depth_image[:, :640, :], cv2.COLOR_BGR2GRAY)
    else:
       gray_depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    
    _, otsu_depth_image = cv2.threshold(
        gray_depth_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    observed_regions = np.logical_and(
                                      np.logical_and(point_radii > 0.03, point_radii < 0.09), 
                                      np.logical_and(otsu_depth_image,
                                                     self.mask_image
                                                    )
                                     )
    observed_binary_image = observed_regions.astype(np.uint8) * 255

    # Perform connected components analysis with statistics
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(observed_binary_image, 
                                                                    connectivity=8)

    # observed_regions_points = np.argwhere(observed_regions)
    # print(np.min(point_radii), np.max(point_radii))
    
    if (num_labels - 1) > 0:
        closest_distance =  np.min(point_radii[observed_regions])
        closest_distance_idx = np.argwhere(point_radii == closest_distance)[0]
        closest_point = point_clound[:, closest_distance_idx[0], closest_distance_idx[1]]
        # repulsive_vector_magnitude = self.Vmax/(1+np.exp(self.alpha*closest_distance*2/self.dmax-self.alpha))        

        # Extract the region where the close point lying in
        target_label = labels[closest_distance_idx[0], closest_distance_idx[1]]
        # Extract the area of each connected component
        areas = stats[1:, cv2.CC_STAT_AREA]
        target_area = areas[target_label-1]

        risk_score = np.log10(target_area * (0.08 / closest_distance))

        # print("Collision Warn")            
        # print("Area: {}".format(target_area))
        # print("Closest distance: {}".format(closest_distance))
        # print("Risk score: {}".format(risk_score))

        # Mark the possible closest point        
        cv2.circle(depth_image, 
                tuple([closest_distance_idx[1], 
                       closest_distance_idx[0]]), 
                radius = 5, 
                color=(255, 0, 0), 
                thickness = -1)
    
    else:
        risk_score = 0.
        closest_distance = np.inf
        closest_point = np.array([np.inf, np.inf, np.inf])

    # for point in observed_regions_points:
    #     cv2.circle(depth_image, 
    #             tuple([point[1], point[0]]), 
    #             radius=2, 
    #             color=(0, 255, 0), 
    #             thickness=-1)
        
        # cv2.putText(
        #         display,  "{0:.2f}".format(closest_rel_depth),
        #         tuple([point[1], point[0]]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
        #         (255, 0, 0)
        #     )


    observed_regions = cv2.cvtColor(observed_regions
                                    .astype(np.uint8) * 255,
                                    cv2.COLOR_GRAY2RGB)
    

    if hasattr(self, "video_shower"):        
        display_image = np.concatenate((observed_regions, depth_image), axis=1)
        self.video_shower.frame = display_image

    if self.frames is not None:
        self.frames.append(depth_image)

    self.previous_fps.append(int(1 / (time.time() - loop_start_time)))

    return risk_score, closest_distance, closest_point, observed_regions, depth_image

def main():    
    pass
    # cam = osc.CameraControl(cam_id=1, 
    #                         exposure_mode="manual",
    #                         exposure_value=-5)
    # cam.set_brightness(0)
    # cam.set_contrast(32)
    # optical_control = osc.OpticalSkinControl(cam, 
    #                                 pdlc_serial_port='COM8',  
    #                                 led_serial_port='COM7',
    #                                 pdlc_type="normal")
    # optical_control.set_transparent()

    # depth_prcessor = DepthSensing()
    
    # try:
    #     # Spin the node so the callback function is called.
    #     while cam.isOpened():
    #         rgb_frame = cam.read()
    #         depth_prcessor.processing(rgb_frame)
    # except KeyboardInterrupt:
    #     depth_prcessor.video_shower.stop()
    #     depth_prcessor.get_logger().info('Stop Depth Perception')
    # except Exception as e:
    #     depth_prcessor.video_shower.stop()
    #     raise e
  
if __name__ == '__main__':
  main()