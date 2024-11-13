"""Processing routine of the Camera Images"""
from abc import ABC, abstractmethod
import numpy as np
import torch

import cv2
import util.image_processing_tools as it


class BaseTacImageProcessor(ABC):
    """This class is an abstract base class (ABC) 
    for tactile image processors
    To create a subclass, you need to impement the following functions:
        -- <__init__>:  initialize the class; first call BaseTacImageProcessor.__init__(self, args)
        -- <process>:   tactile image processing function
    """

    def __init__(
        self,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256)
    ):
        assert isinstance(cropped_size, tuple)
        assert isinstance(resized_size, tuple)

        self.cropped_size = cropped_size
        self.resized_size = resized_size

    def __call__(self, sample: np.ndarray):
        image = self.process(sample)
        return self.__img2tensor(image).unsqueeze(0)

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process tactile images"""
        pass

    @staticmethod
    def __img2tensor(img: np.ndarray) -> torch.Tensor:
        """Method for translating image into tensor"""
        img = (img / 255.0).astype(np.float32)
        return torch.from_numpy(img)


class BinaryTacImageProcessorCoverMask(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        mask_radius: int = 180,
        apply_block_mask: int = True,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self, 
                            cropped_size, 
                            resized_size)

        self.threshold = threshold
        self.filter_size = filter_size
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)
        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.apply_mask_to_image(image, self.mask_radius)
        image = it.resize_image(image, self.resized_size)        
        if self.apply_block_mask:
            image = it.apply_invert_mask_to_image(
                                image, 
                                radius = self.block_mask_radius, 
                                center = self.block_mask_center)
        
        return image


class BinaryTacImageProcessor(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        block_mask_radius: int = 20,
        block_mask_center: tuple = (128, 128)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self, 
                            cropped_size, 
                            resized_size)

        self.threshold = threshold
        self.filter_size = filter_size
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)
        image = it.crop_image_at_center(image, self.cropped_size)
        image = it.resize_image(image, self.resized_size)
        
        image = it.apply_invert_mask_to_image(image, 
                                              radius = self.block_mask_radius, 
                                              center = self.block_mask_center)

        return image


class WrapedBinaryTacImageProcessor(BaseTacImageProcessor):
    """Class for handling the image transforms"""

    def __init__(
        self,
        homography: np.ndarray,
        threshold: int = 42,
        filter_size: int = 3,
        cropped_size: tuple = (400, 400),
        resized_size: tuple = (256, 256),
        apply_mask = False,
        mask_radius: int = 180,
        apply_block_mask = True,
        block_mask_radius: int = 27,
        block_mask_center: tuple = (131, 120)
    ):
        assert isinstance(threshold, int)
        assert isinstance(filter_size, int)
        assert isinstance(cropped_size, (int, tuple))
        assert isinstance(resized_size, (int, tuple))

        BaseTacImageProcessor.__init__(
                            self, 
                            cropped_size, 
                            resized_size)

        self.homography = homography
        self.threshold = threshold
        self.filter_size = filter_size
        self.apply_mask = apply_mask
        self.mask_radius = mask_radius
        self.apply_block_mask = apply_block_mask
        self.block_mask_radius = block_mask_radius
        self.block_mask_center = block_mask_center

    # comment this function if you want to use
    # the one of abstract ABS class
    # def __call__(self, sample: np.ndarray):
    #     image = self.process(sample)
    #     return image

    def process(self, image: np.ndarray):
        image = it.rbg_to_grayscale(image)
        image = it.blur_image(image, self.filter_size)
        image = it.create_binary_image(image, self.threshold)
        image = it.crop_image_at_center(image, self.cropped_size)
        image = cv2.warpPerspective(image, 
                                    self.homography, 
                                    image.shape)
        image = it.resize_image(image, self.resized_size)  
        if self.apply_mask:
            image = it.apply_mask_to_image(image, self.mask_radius)        
        if self.apply_block_mask: 
            image = it.apply_invert_mask_to_image(image, 
                                                radius = self.block_mask_radius, 
                                                center = self.block_mask_center)
        return image