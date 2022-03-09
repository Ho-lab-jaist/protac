import cv2
import torch
from torchvision import transforms

import os
import numpy as np
import pandas as pd

from .dlmodel import TacNet

def get_free_node_ind(node_idx_path, label_idx_path):
    df_node_idx = pd.read_csv(node_idx_path)
    df_label_idx = pd.read_csv(label_idx_path)

    node_idx = np.array(df_node_idx.iloc[:,0], dtype=int) # (full skin) face node indices in vtk file exported from SOFA 
    node_idx = list(set(node_idx)) # eleminate duplicate elements (indices)
    node_idx = sorted(node_idx) # sorted the list of indices

    label_idx = list(df_label_idx.iloc[:,0]) #(not full skin) at nodes used for training - labels
    file_idx = [node_idx.index(idx) for idx in label_idx]

    return file_idx

full_path = '/home/protac/ros/protac_ws/src/protac_control/resource'
class TactilePerception(object):
    def __init__(self, tacnet_dir=full_path,
                       trained_model = 'TacNet_Unet_real_data.pt',
                       cam_ind = [0, 2],
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
        touched_nodes_depth = nodes_depth[(nodes_depth > 4.5)]
        # the number of touched nodes
        num_of_touched_nodes = len(touched_nodes_depth)
        return True if num_of_touched_nodes > 2 else False

if __name__ == "__main__":
    tacitle_perception = TactilePerception()
    while True:
        if tacitle_perception.detect_contact():
            print('Contact')
        else:
            print('No')