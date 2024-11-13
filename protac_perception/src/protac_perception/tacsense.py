import numpy as np
import torch
import os
import pandas as pd

from protac_perception.model.tacnet_model import TacNet
from protac_perception.util.processing import WrapedBinaryTacImageProcessor
import rospkg

r = rospkg.RosPack()
path = r.get_path('protac_perception')
FULL_PATH = path + '/resource'

# PDLC type: normal
homography = np.array([[0.8963773668, -0.0218227057, 16.2641985261],
                       [-0.0243724548, 0.8934696105, 21.1218481758],
                       [-0.0000691098, -0.0001396142, 1]])

# PDLC type: reverse
# homography = np.array([[0.9114095007, -0.045434745, 29.7169054886],
#                        [0.0110005951, 0.8912473111, 22.0472590743],
#                        [-0.0000212762, -0.0001429191, 1]])

transform = WrapedBinaryTacImageProcessor(
                                        homography = homography,
                                        threshold = 38,
                                        filter_size = 5,
                                        cropped_size  = (450, 450),
                                        resized_size  = (256, 256),
                                        apply_mask = True,
                                        mask_radius = 125,
                                        apply_block_mask = True,
                                        block_mask_radius = 24,
                                        block_mask_center = (128, 128))

class TactileProcessor():
    def __init__(self, 
                 num_of_nodes,
                 model_file,
                 transform = transform,
                 model_type = TacNet,
                ):

        self.transform = transform
        self.first_step = True
        self.num_of_nodes = num_of_nodes
        self.first_defomration = np.zeros(self.num_of_nodes)

        INIT_PATH = os.path.join(FULL_PATH, 'init_pos.csv')  
        init_pos_csv = pd.read_csv(INIT_PATH)
        self.init_positions = (np.array(init_pos_csv.iloc[0, 1:], dtype=float)
                               .reshape(-1, 3))
        
        # Load TacNet
        MODEL_PATH = os.path.join(FULL_PATH, model_file)
        self.tacnet = model_type(in_nc = 1, num_of_features = num_of_nodes)
        print('model [TacNet] was created')
        print('loading the model from {0}'.format(MODEL_PATH))
        self.tacnet.load_state_dict(torch.load(MODEL_PATH))
        print('---------- Tactile Networks initialized -------------')
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.dev)
        self.tacnet.to(self.dev)
        self.tacnet.eval()

    def set_input(self, rgb_img):
        img = transform(rgb_img)
        self.input_img = img.unsqueeze(0).to(self.dev)
    
    def get_estimated_deformation(self):
        return self._inference()
    
    def _inference(self):
        with torch.no_grad():
            # predict of skin deformation
            estimated_deformation = (self.tacnet(self.input_img)
                                     .cpu()
                                     .numpy()
                                     .reshape(-1, 3))/5 
        
        return (estimated_deformation, 
                self._post_processing_data(np.linalg.norm(estimated_deformation, axis=1)))

    def _post_processing_data(self, estimated_deformation):
        deformation = np.zeros(self.num_of_nodes)
        if self.first_step:
            self.first_defomration = estimated_deformation
            self.first_step = False
        
        deformation = estimated_deformation

        return np.clip(deformation - self.first_defomration, 0, 1000)

    """
    Sensor Processing Method for Events
    """
    def extract_contact_area(self):
        """
        Return node indices where contact is made,
        and the corresponding node depth (displacement intensity at the node)
        """
        nodes_depth = self.get_estimated_deformation()
        touched_nodes_indices = np.where(nodes_depth > 2)[0]
        return nodes_depth[touched_nodes_indices], touched_nodes_indices

    def detect_touch(self):
        """
        Trigger an event by True signal when an contact occurs on the skin
        Binary classification task
        """
        nodes_depth = self.get_estimated_deformation()
        # extract nodes where depth > epsilon = 5 mm
        touched_nodes_depth = nodes_depth[(nodes_depth > 2)]
        # the number of touched nodes
        num_of_touched_nodes = len(touched_nodes_depth)
        return True if num_of_touched_nodes > 5 else False