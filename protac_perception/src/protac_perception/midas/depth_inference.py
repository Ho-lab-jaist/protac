"""
Author: Quan Khanh Luu
Email: quan-luu@jaist.ac.jp
Description: Run monocular depth inference realtime on "protac" images using pre-trained MiDas model
References for MiDas:
Github: https://github.com/isl-org/MiDaS
Paper: https://arxiv.org/abs/1907.01341
"""

import argparse
import os
import threading
import numpy as np

import torch
from torchvision.transforms import Compose
import cv2

from .dpt_depth import DPTDepthModel
from .midas_net import MidasNet
from .midas_net_custom import MidasNet_small
from .transforms import Resize, NormalizeImage, PrepareForNet

class VideoGetter():
    def __init__(self, src=0):
        """
        Class to read frames from a VideoCapture in a dedicated thread.

        Args:
            src (int|str): Video source. Int if webcam id, str if path to file.
        """
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        Method called in a thread to continually read frames from `self.cap`.
        This way, a frame is always ready to be read. Frames are not queued;
        if a frame is not read before `get()` reads a new frame, previous
        frame is overwritten.
        """
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()

    def stop(self):
        self.stopped = True

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


def encode_depth(depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    # Uncomment below for saving .pfm extension
    # write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * ((depth - depth_min) / (depth_max - depth_min))
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits==1:
        return out.astype(np.uint8)
    else:
        return out.astype(np.uint16)


def write_mp4(frames, fps, filepath):
    """
    Write provided frames to an .mp4 video.

    Args:
        frames (list): List of frames (np.ndarray).
        fps (int): Framerate (frames per second) of the output video.
        filepath (str): Path to output video file.
    """
    if not filepath.endswith(".mp4"):
        filepath += ".mp4"

    h, w = frames[0].shape[:2]

    # writer = cv2.VideoWriter(
    #     filepath, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h), False
    # )

    writer = cv2.VideoWriter(
        filepath, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h), True
    )

    for frame in frames:
        writer.write(frame)
    writer.release()


class DepthProcessing():
    def __init__(self, model_weights=None, model_type="dpt_large", optimize=True):
        
        if model_weights is None:
            default_models = {
                "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
                "midas_v21": "weights/midas_v21-f6b98070.pt",
                "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
                "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
                }
        else:
            default_models = {
                "midas_v21_small": os.path.join(model_weights, "midas_v21_small-70d6b9c8.pt"),
                "midas_v21": os.path.join(model_weights, "midas_v21-f6b98070.pt"),
                "dpt_large": os.path.join(model_weights, "dpt_large-midas-2f21e586.pt"),
                "dpt_hybrid": os.path.join(model_weights, "dpt_hybrid-midas-501f0c75.pt"),
                }
        
        self.model_type = model_type

        self.model_weights = default_models[self.model_type]

        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        # load network
        if self.model_type == "dpt_large": # DPT-Large
            self.model = DPTDepthModel(
                path=self.model_weights,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.model_type == "dpt_hybrid": #DPT-Hybrid
            self.model = DPTDepthModel(
                path=self.model_weights,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode="minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.model_type == "midas_v21":
            self.model = MidasNet(
                path=self.model_weights, 
                non_negative=True)
            net_w, net_h = 384, 384
            resize_mode="upper_bound"
            normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.model_type == "midas_v21_small":
            self.model = MidasNet_small(
                path=self.model_weights, 
                features=64, 
                backbone="efficientnet_lite3", 
                exportable=True, 
                non_negative=True, 
                blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode="upper_bound"
            normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            print(f"model_type '{self.model_type}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.optimize=optimize

        self.model.eval()
        
        if self.optimize==True:
            # rand_example = torch.rand(1, 3, net_h, net_w)
            # model(rand_example)
            # traced_script_module = torch.jit.trace(model, rand_example)
            # model = traced_script_module
        
            if self.device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)  
                self.model = self.model.half()

        self.model.to(self.device)


        # Intrinsic camera calibration matrix
        self.K = np.array([[239.35312993382436, 0.00000000000000, 308.71493813687908],
                           [0.00000000000000, 239.59440270542146, 226.24771387864561],
                           [0.00000000000000, 0.00000000000000, 1.00000000000000]])

        # Create a 2D image gird points
        x_arr = np.linspace(0, 639, 640)
        y_arr = np.linspace(0, 479, 480)
        self.X_img, self.Y_img = np.meshgrid(x_arr, y_arr)
        
        
    def run(self, frame):
        """ Run depth prediction and return raw depth distance
        """
        # input
        self.frame=frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        img_input = self.transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize==True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            self.prediction = prediction
        return prediction

    def encode_depth(self, bits=1):
        """Encode depth (distance) map to image format.
        """
        # Uncomment below for saving .pfm extension
        # write_pfm(path + ".pfm", depth.astype(np.float32))
        
        depth_min = self.prediction.min()
        depth_max = self.prediction.max()

        max_val = (2**(8*bits))-1
        
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * ((self.prediction - depth_min) / (depth_max - depth_min))
        else:
            out = np.zeros(self.prediction.shape, dtype=self.prediction.type)

        if bits==1:
            return out.astype(np.uint8)
        else:
            return out.astype(np.uint16)

    def display_depth(self, bits=1, fps=None, raw_append=True):
        # encode the depth map predection for display
        encoded_depth = self.encode_depth(bits=bits)
        
        # put fps on the depth stream
        if fps is not None:
            cv2.putText(
                encoded_depth,  f"{fps} fps",
                (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (255, 255, 255)
            )
        if raw_append:
            return np.concatenate((cv2.cvtColor(encoded_depth, cv2.COLOR_GRAY2RGB), self.frame), axis=1)
        else:
            return cv2.cvtColor(encoded_depth, cv2.COLOR_GRAY2RGB)

    def disparity2depth(self, disparity, dsp_const):
        """ Convert disparity to depth
        - Parameters:
        @disparity (float): disparity of the target point
        @dsp_const (float): disparity constant (f*b)

        - Returns:
        (float): depth of the target point    
        """
        self.depth = dsp_const/disparity
        return self.depth

    def back_projection(self, depth, img_point=None):
        """ Compute 3D point from the image point (pixel) and depth information (metrics)
        - Parameters:
        @img_point (nd.array; (2,)): the target point on image plane (x, y)
        @depth (float): the estimated depth
        @K (nd.array): the intrinsic calibration matrix
        - Returns:
        (nd.array; (3,)): 3D coordinate of the target point w.r.t the camera reference frame {C}
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        u0 = self.K[0, 2]
        v0 = self.K[1, 2]

        if img_point is None:
            u = self.X_img
            v = self.Y_img 
        else:
            u = img_point[0]
            v = img_point[1]

        x = np.multiply(depth, (u-u0)/fx)
        y = np.multiply(depth, (v-v0)/fy)

        return np.array([x, y, depth])


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--cam_index', 
    #     default=0,
    #     help='camera index'
    # )

    # parser.add_argument('-o', '--output_path', 
    #     default='output/test_depth.mp4',
    #     help='folder for output images'
    # )

    # parser.add_argument('-m', '--model_weights', 
    #     default=None,
    #     help='path to the trained weights of model'
    # )

    # parser.add_argument('-t', '--model_type', 
    #     default='dpt_hybrid',
    #     help='model type: dpt_large, dpt_hybrid, midas_v21 or midas_v21_small'
    # )

    # parser.add_argument('--optimize', dest='optimize', action='store_true')
    # parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    # parser.set_defaults(optimize=True)

    # args = parser.parse_args()

    # default_models = {
    #     "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
    #     "midas_v21": "weights/midas_v21-f6b98070.pt",
    #     "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    #     "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    # }

    # if args.model_weights is None:
    #     args.model_weights = default_models[args.model_type]

    # # set torch options
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    # # select device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: %s" % device)

    # cam_idx = args.cam_index
    # output_path = args.output_path
    # model_type = args.model_type
    # model_path = args.model_weights
    # optimize = args.optimize

    # camera_mat = np.loadtxt('K_fisheye.csv', delimiter=',')
    # dist_coef = np.loadtxt('d_fisheye.csv', delimiter=',')
    # k_new_param = 0.9
    # new_camera_mat = camera_mat.copy()
    # new_camera_mat[(0, 1), (0, 1)] = k_new_param * new_camera_mat[(0, 1),
    #                                                               (0, 1)]

    # # load network
    # if model_type == "dpt_large": # DPT-Large
    #     model = DPTDepthModel(
    #         path=model_path,
    #         backbone="vitl16_384",
    #         non_negative=True,
    #     )
    #     net_w, net_h = 384, 384
    #     resize_mode = "minimal"
    #     normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # elif model_type == "dpt_hybrid": #DPT-Hybrid
    #     model = DPTDepthModel(
    #         path=model_path,
    #         backbone="vitb_rn50_384",
    #         non_negative=True,
    #     )
    #     net_w, net_h = 384, 384
    #     resize_mode="minimal"
    #     normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # elif model_type == "midas_v21":
    #     model = MidasNet(model_path, non_negative=True)
    #     net_w, net_h = 384, 384
    #     resize_mode="upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    # elif model_type == "midas_v21_small":
    #     model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
    #     net_w, net_h = 256, 256
    #     resize_mode="upper_bound"
    #     normalization = NormalizeImage(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    # else:
    #     print(f"model_type '{model_type}' not implemented, use: --model_type large")
    #     assert False
    
    # transform = Compose(
    #     [
    #         Resize(
    #             net_w,
    #             net_h,
    #             resize_target=None,
    #             keep_aspect_ratio=True,
    #             ensure_multiple_of=32,
    #             resize_method=resize_mode,
    #             image_interpolation_method=cv2.INTER_CUBIC,
    #         ),
    #         normalization,
    #         PrepareForNet(),
    #     ]
    # )

    # model.eval()
    
    # if optimize==True:
    #     # rand_example = torch.rand(1, 3, net_h, net_w)
    #     # model(rand_example)
    #     # traced_script_module = torch.jit.trace(model, rand_example)
    #     # model = traced_script_module
    
    #     if device == torch.device("cuda"):
    #         model = model.to(memory_format=torch.channels_last)  
    #         model = model.half()

    # model.to(device)

    # start_time = time.time()

    # # Wrap in try/except block so that output video is written
    # # even if an error occurs while streaming webcam input.
    # try:
    #     # start video stream
    #     video_getter = VideoGetter(cam_idx).start()
    #     video_shower = VideoShower(None, "Depth live").start()
    #     # video_shower_rgb = VideoShower(None, "RGB live").start()

    #     # create saved frames
    #     frames = list()

    #     # Number of frames to average for computing FPS.
    #     num_fps_frames = 30
    #     previous_fps = deque(maxlen=num_fps_frames)
        
    #     while True:
    #         loop_start_time = time.time()

    #         if video_getter.stopped or video_shower.stopped:
    #             video_getter.stop()
    #             video_shower.stop()
    #             break

    #         original_frame = video_getter.frame
    #         frame = cv2.fisheye.undistortImage(
    #             original_frame,
    #             camera_mat,
    #             D=dist_coef,
    #             Knew=new_camera_mat,
    #         )

    #         # input
    #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    #         img_input = transform({"image": img})["image"]

    #         # compute
    #         with torch.no_grad():
    #             sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
    #             if optimize==True and device == torch.device("cuda"):
    #                 sample = sample.to(memory_format=torch.channels_last)  
    #                 sample = sample.half()
    #             prediction = model.forward(sample)
    #             prediction = (
    #                 torch.nn.functional.interpolate(
    #                     prediction.unsqueeze(1),
    #                     size=img.shape[:2],
    #                     mode="bicubic",
    #                     align_corners=False,
    #                 )
    #                 .squeeze()
    #                 .cpu()
    #                 .numpy()
    #             )

    #         # encode the depth predection for display
    #         depth = encode_depth(prediction, bits=1)

    #         # put fps on the depth stream
    #         cv2.putText(
    #             depth,  f"{int(sum(previous_fps) / num_fps_frames)} fps",
    #             (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
    #             (255, 255, 255)
    #         )

    #         previous_fps.append(int(1 / (time.time() - loop_start_time)))

    #         display = np.concatenate((cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB), frame), axis=1)
    #         video_shower.frame = display

    #         if frames is not None:
    #             frames.append(display)

    # except Exception as e:
    #     raise e
    # finally:
    #     if output_path and frames:  
    #         # Get average FPS and write output at that framerate.
    #         fps = 1 / ((time.time() - start_time) / len(frames))
    #         write_mp4(frames, fps, output_path)