import cv2
import torch
from torchvision import transforms

import os
import numpy as np
import pandas as pd

from PIL import Image

from .dlmodel import TacNet, ForceNet
from .mcl import mcl

def get_free_node_ind(node_idx_path, label_idx_path):
    df_node_idx = pd.read_csv(node_idx_path)
    df_label_idx = pd.read_csv(label_idx_path)

    node_idx = np.array(df_node_idx.iloc[:,0], dtype=int) # (full skin) face node indices in vtk file exported from SOFA 
    node_idx = list(set(node_idx)) # eleminate duplicate elements (indices)
    node_idx = sorted(node_idx) # sorted the list of indices

    label_idx = list(df_label_idx.iloc[:,0]) #(not full skin) at nodes used for training - labels
    file_idx = [node_idx.index(idx) for idx in label_idx]

    return file_idx

full_path = '/home/protac/ros/protac_ws/src/protac_perception/resource'

class TactilePerception(object):
    def __init__(self, tacnet_dir = full_path,
                       trained_model = 'TacNet_Unet_real_data_normalized_220623.pt',
                       cam_ind = [0, 8],
                       num_of_nodes = 707,
                       node_idx_path=os.path.join(full_path,'node_idx.csv'), 
                       label_idx_path=os.path.join(full_path,'label_idx.csv')):

        # Soft skin representation
        self.num_of_nodes = num_of_nodes
        self.free_node_ind = get_free_node_ind(node_idx_path, label_idx_path)

        # Initialize TacNet
        self.model_dir = os.path.join(tacnet_dir, trained_model)
        self.init_TacNet()

        # Initialize Cameras
        """
        For sucessfully read two video camera streams simultaneously,
        we need to use two seperated USB bus for cameras
        e.g, USB ports in front and back of the CPU
        """
        self.cam_bot = cv2.VideoCapture(cam_ind[0])
        self.cam_top = cv2.VideoCapture(cam_ind[1])
        if self.cam_bot.isOpened() and self.cam_top.isOpened():
            print('Cameras are ready!')
        else:
            assert False, 'Camera connection failed!'

    def init_TacNet(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tacnet = TacNet()
        print('[Tacnet] model was created')
        self.print_networks(False)
        print('loading the model from {0}'.format(self.model_dir))
        self.tacnet.load_state_dict(torch.load(self.model_dir))
        print('---------- TacNet initialized -------------')
        self.tacnet.to(self.dev)
        self.tacnet.eval()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        net = getattr(self, 'tacnet')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[TacNet] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def estimate_skin_deformation(self):
        """
        Return the estimate of skin node's displacements X - X0 (N, 3)
        N is the number of nodes, and 3 is Dx, Dy, Dz
        Refer to Section 3. (Skin Deformation Estimation) in the paper
        """
        self.update_tactile_images()
        self._free_node_displacments = self.estimate_free_node_displacments()
        return self.get_full_node_displacments()

    def update_tactile_images(self):
        # Read marker-featured tactile images from camera video streams
        frame_top = cv2.cvtColor(self.cam_top.read()[1], cv2.COLOR_BGR2RGB)
        frame_bot = cv2.cvtColor(self.cam_bot.read()[1], cv2.COLOR_BGR2RGB)
        # Apply pre-processing to the pair of tactile images
        self._frame_top = self.apply_transform(frame_top)
        self._frame_bot = self.apply_transform(frame_bot)
        # Concantenate the two tactile images
        self._tac_img = torch.cat((self._frame_top, self._frame_bot), dim=1).to(self.dev)

    def estimate_free_node_displacments(self):
        with torch.no_grad():
            node_displacments = self.tacnet(self._tac_img).cpu().numpy()
            return node_displacments

    def get_full_node_displacments(self):
        """
        The full skin deformation includes deviations of fixed nodes which is zero, 
        and the free nodes calculated in "estimate" function
        """
        
        displacements = np.zeros((self.num_of_nodes, 3))
        displacements[self.free_node_ind, :] = self._free_node_displacments.reshape(-1, 3)

        return displacements

    def apply_transform(self, img):
        """
        Apply pre-processing for the inputted image
        Parameters:
            img: image in numpy nd.array (C, H, W)
        Returns:
            processed image in tensor format (1, C, H, W)
        """
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(480),
        transforms.Resize((256,256))
        ])
        return transform(img).unsqueeze(0)

    """
    Sensor Processing Method for Events
    """
    def extract_contact_area(self):
        """
        Return node indices where contact is made,
        and the corresponding node depth (displacement intensity at the node)
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        touched_nodes_indices = np.where(nodes_depth > 2.5)[0]
        return nodes_depth[touched_nodes_indices], touched_nodes_indices

    def detect_contact(self):
        """
        Trigger an event by True signal when an contact occurs on the skin
        Binary classification task
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        # extract nodes where depth > epsilon = 5 mm
        touched_nodes_depth = nodes_depth[(nodes_depth > 2)]
        # the number of touched nodes
        num_of_touched_nodes = len(touched_nodes_depth)
        return True if num_of_touched_nodes > 2 else False

    def extract_contact_positions(self):
        """
        Extract the contact positions acting on the TacLink
        """
        contact_radial_vectors, contact_positions, contact_depths = mcl(self.estimate_skin_deformation(), 2)
        return contact_positions, contact_radial_vectors, contact_depths

class DeformationSensing(object):        
    def __init__(self, tacnet_dir = full_path,
                       trained_model = 'TacNet_Unet_real_data_normalized_220623.pt',
                       cam_ind = [0, 8],
                       num_of_nodes = 707,
                       node_idx_path=os.path.join(full_path,'node_idx.csv'), 
                       label_idx_path=os.path.join(full_path,'label_idx.csv')):

        # Soft skin representation
        self.num_of_nodes = num_of_nodes
        self.free_node_ind = get_free_node_ind(node_idx_path, label_idx_path)

        # Initialize TacNet
        self.model_dir = os.path.join(tacnet_dir, trained_model)
        self.init_TacNet()

        # initialize contact information
        self.contact_radial_vectors = None
        self.contact_positions = None
        self.contact_depths = None

        # Initialize Cameras
        """
        For sucessfully read two video camera streams simultaneously,
        we need to use two seperated USB bus for cameras
        e.g, USB ports in front and back of the CPU
        """
        self.cam_bot = cv2.VideoCapture(cam_ind[0])
        self.cam_top = cv2.VideoCapture(cam_ind[1])
        if self.cam_bot.isOpened() and self.cam_top.isOpened():
            print('Cameras are ready!')
        else:
            assert False, 'Camera connection failed!'

        # define affine transformation matrix
        self.M_top = np.float32([[1, 0, -4],
                                 [0, 1, 0]])
        self.M_bot = np.float32([[1, 0, -4.5],
                                 [0, 1, +1.9]])

    def init_TacNet(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tacnet = TacNet()
        print('[Tacnet] model was created')
        self.print_networks(False)
        print('loading the model from {0}'.format(self.model_dir))
        self.tacnet.load_state_dict(torch.load(self.model_dir))
        print('---------- TacNet initialized -------------')
        self.tacnet.to(self.dev)
        self.tacnet.eval()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        net = getattr(self, 'tacnet')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[ForceNet] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def apply_transform(self, img, mean_dataset=None, std_dataset=None):
        tf_list =  [transforms.CenterCrop(480),
                    transforms.Resize((256,256)),
                    transforms.ToTensor()]
        if mean_dataset != None and std_dataset != None:
            tf_list.append(transforms.Normalize(mean_dataset, std_dataset))
        transform = transforms.Compose(tf_list)

        return transform(img).unsqueeze(0)

    def update_tactile_images(self):
        # Read marker-featured tactile images from camera video streams
        frame_top0 = cv2.cvtColor(self.cam_top.read()[1], cv2.COLOR_BGR2RGB)
        frame_bot0 = cv2.cvtColor(self.cam_bot.read()[1], cv2.COLOR_BGR2RGB)
        # performance affine transformation on the original images
        frame_top = cv2.warpAffine(frame_top0, self.M_top, (frame_top0.shape[1], frame_top0.shape[0]))
        frame_bot = cv2.warpAffine(frame_bot0, self.M_bot, (frame_bot0.shape[1], frame_bot0.shape[0]))
        # convert numpy to PIL images for consistency with training dataloader
        frame_top_PIL = Image.fromarray(frame_top)
        frame_bot_PIL = Image.fromarray(frame_bot)
        # Apply pre-processing to the pair of tactile images
        self._frame_top = self.apply_transform(frame_top_PIL, (0.1096, 0.1031, 0.1166), (0.1238, 0.1120, 0.1011))
        self._frame_bot = self.apply_transform(frame_bot_PIL, (0.1512, 0.1159, 0.1349), (0.1669, 0.1184, 0.1151))
        # Concantenate the two tactile images
        self._tac_img = torch.cat((self._frame_top, self._frame_bot), dim=1).to(self.dev)

    def estimate_free_node_displacments(self):
        with torch.no_grad():
            node_displacments = self.tacnet(self._tac_img).cpu().numpy()
            return node_displacments

    def get_full_node_displacments(self):
        """
        The full skin deformation includes deviations of fixed nodes which is zero, 
        and the free nodes calculated in "estimate" function
        """
        self.update_tactile_images()
        self._free_node_displacments = self.estimate_free_node_displacments()
        displacements = np.zeros((self.num_of_nodes, 3))
        displacements[self.free_node_ind, :] = self._free_node_displacments.reshape(-1, 3)

        return displacements

    def estimate_skin_deformation(self):
        """
        Return the estimate of skin node's displacements X - X0 (N, 3)
        N is the number of nodes, and 3 is Dx, Dy, Dz
        Refer to Section 3. (Skin Deformation Estimation) in the paper
        """
        return self.get_full_node_displacments()

    """
    Sensor Processing Method for Events
    """
    def extract_contact_area(self):
        """
        Return node indices where contact is made,
        and the corresponding node depth (displacement intensity at the node)
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        touched_nodes_indices = np.where(nodes_depth > 2.5)[0]
        return nodes_depth[touched_nodes_indices], touched_nodes_indices

    def detect_contact(self):
        """
        Trigger an event by True signal when an contact occurs on the skin
        Binary classification task
        """
        full_node_displacements = self.estimate_skin_deformation()
        nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
        # extract nodes where depth > epsilon = 5 mm
        touched_nodes_depth = nodes_depth[(nodes_depth > 2)]
        # the number of touched nodes
        num_of_touched_nodes = len(touched_nodes_depth)
        return True if num_of_touched_nodes > 2 else False

    def extract_contact_information(self):
        """
        Extract the contact information of interest acting on the TacLink,
        computed from MCL (multi-contact labeling) module
        """
        contact_radial_vectors, contact_positions, contact_depths = mcl(self.estimate_skin_deformation(), 3)
        return contact_depths, contact_positions, contact_radial_vectors 

class ForceSensing(object):
    def __init__(self, forcenet_dir = full_path,
                       vtk_file = 'protacSkin.vtk',
                       trained_model = 'ForceNet_Unet_RGB_onecame_220620(Force).pt'):

        # Initialize ForceNet
        self.model_dir = os.path.join(forcenet_dir, trained_model)
        self.vtk_dir = os.path.join(forcenet_dir, vtk_file)
        self.init_ForceNet()
        self.init_positions = self.extract_skin_points()
        self.estimated_force_map = None
        
    def extract_skin_points(self):
        points_ls = []
        with open(self.vtk_dir, 'r') as rf:
            for idx, line in enumerate(rf):
                if 6 <= idx <= 626:
                    points = [float(x) for x in line.split()]
                    points_ls.append(points)
                elif idx > 626:
                    break
        return np.array(points_ls, dtype=float)

    def init_ForceNet(self):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.forcenet = ForceNet()
        print('[ForceNet] model was created')
        self.print_networks(False)
        print('loading the model from {0}'.format(self.model_dir))
        self.forcenet.load_state_dict(torch.load(self.model_dir))
        print('---------- ForceNet initialized -------------')
        self.forcenet.to(self.dev)
        self.forcenet.eval()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        net = getattr(self, 'forcenet')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[ForceNet] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def run(self, tac_image):
        tac_image = self.apply_transform(tac_image)
        input_img = tac_image.to(self.dev)
        # forward pass to ForceNet
        with torch.no_grad():
            # estimation of force map
            self.estimated_force_map = self.forcenet(input_img).cpu().numpy().squeeze(0)

    def apply_transform(self, img):
        """
        Apply pre-processing for the inputted image (PIL format)
        Parameters:
            img: image in numpy nd.array (C, H, W)
        Returns:
            processed image in tensor format (1, C, H, W)
        """
        transform = transforms.Compose([
        transforms.CenterCrop(480),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.7417, 0.7580, 0.7576), (0.1145, 0.1193, 0.1222)),
        ])
        return transform(img).unsqueeze(0)

    def forcemap2vector(self, force_map):
        """
        Convert 3D force map (3, H, W) to an array of distributed force vector (3, -1)
        """
        # return np.vstack((force_map[0].flatten(), force_map[1].flatten(), force_map[2].flatten())).transpose()
        # denormalize force data
        force_map[0] = (force_map[0]*0.1133)+1.1290e-03
        force_map[1] = (force_map[1]*0.1174)+-3.5948e-04
        return np.vstack((force_map[0].flatten(), force_map[1].flatten())).transpose()

    """
    Sensor Processing Method for Events
    """
    def compute_force_map(self, tac_image):
        """
        Return estimated 3D force map
        """
        self.run(tac_image)
        return self.estimated_force_map

    def compute_contact_force(self, tac_image):
        """
        Extract the contact forces and positions acting on the ProTac link
        """
        # run force-map inference
        self.run(tac_image)
        # convert to force vector
        estimated_focrce_vector = self.forcemap2vector(self.estimated_force_map)
        intensity_force_vector = np.linalg.norm(estimated_focrce_vector, axis=1)
        single_contact_force = np.max(intensity_force_vector)
        contact_location_id = np.where(intensity_force_vector==single_contact_force)[0].squeeze()
        # 3D-coordinated contact position of given contact region
        single_contact_position = self.init_positions[contact_location_id]
        return single_contact_force, single_contact_position

    # def detect_contact(self):
    #     """
    #     Trigger an event by True signal when an contact occurs on the skin
    #     Binary classification task
    #     """
    #     full_node_displacements = self.estimate_skin_deformation()
    #     nodes_depth = np.linalg.norm(full_node_displacements, axis=1)
    #     # extract nodes where depth > epsilon = 5 mm
    #     touched_nodes_depth = nodes_depth[(nodes_depth > 2)]
    #     # the number of touched nodes
    #     num_of_touched_nodes = len(touched_nodes_depth)
    #     return True if num_of_touched_nodes > 2 else False

if __name__ == "__main__":
    # tacitle_perception = TactilePerception()    
    # contact_count = 0
    # while True:
    #     contact_positions = tacitle_perception.extract_contact_positions()
    #     if tacitle_perception.detect_contact() and len(contact_positions)==1:
    #         print('Contact: {0}'.format(len(contact_positions)))
    #         contact_count +=1
    #         current_position = contact_positions[0]
    #         if contact_count>=4:
    #             directional_vector = current_position-prev_position
    #             distance = np.linalg.norm(directional_vector)
    #             if distance > 4: # unit: mm
    #                 # print(directional_vector[2])
    #                 if directional_vector[2] > 0:
    #                     sign = "+"
    #                 else:
    #                     sign = "-"
    #                 print("Stroke Detected ({0})".format(sign))
    #             else:
    #                 # print(distance)
    #                 print("Press Detected")
    #         prev_position = current_position
    #     elif tacitle_perception.detect_contact() and len(contact_positions)>=2:
    #         # print('Multi-point contact: {0}'.format(len(contact_positions)))
    #         pass
    #     else:
    #         print('No: {0}'.format(len(contact_positions)))
    #         contact_count = 0

    """
    Test ForceSensing module
    """
    tactile_sensing = ForceSensing()
    pseudo_image_array = np.clip(np.random.rand(480, 640, 3)*255, 0, 255).astype(np.uint8)
    pseudo_image = Image.fromarray(pseudo_image_array)
    contact_force, contact_position = tactile_sensing.compute_contact_force(pseudo_image)
    print(contact_force, contact_position)
