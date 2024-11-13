"""Tools for image processing"""

import cv2
import IPython
import numpy as np
import torch
from skimage.util import img_as_ubyte


def sort_marker_center_points(center_marker_points: np.ndarray, 
                              center_x: int, 
                              center_y: int,
                              n_row: int):
    """Convert coordinates of marker center points from the
    Cartesian to polar coordinates"""
    
    x_coordinate_in_image_center = center_marker_points[:, 0] - center_x
    y_coordinate_in_image_center = center_marker_points[:, 1] - center_y

    radii = np.sqrt(x_coordinate_in_image_center ** 2 + y_coordinate_in_image_center ** 2)
    angles = np.rad2deg(np.arctan2(y_coordinate_in_image_center, x_coordinate_in_image_center))
    data_len = len(radii)

    converted_points = np.concatenate([radii.reshape(data_len, 1), angles.reshape(data_len, 1)], axis=1)
    sorted_indexes_with_radius = converted_points[:, 0].argsort()
    converted_points = converted_points[sorted_indexes_with_radius]
    converted_points = converted_points.reshape(-1, n_row, 2)
    sorted_indexes_with_radius = sorted_indexes_with_radius.reshape(-1, n_row, 1)
    sorted_indexes_with_angles = np.array([indexes[points[:, 1].argsort()] for points, indexes in zip(converted_points, sorted_indexes_with_radius)]).reshape(-1)

    return center_marker_points[sorted_indexes_with_angles]


def remove_duplicate_points(center_points: list, distance_threshold: int = 3) -> list:
    """Remove duplicate 2D points in an array"""
    filtered_center_points = list()
    center_points = np.unique(center_points, axis=0)
    for center_point in center_points:
        if len(filtered_center_points) == 0:
            filtered_center_points.append(center_point)
        else:
            for filtered_point in filtered_center_points:
                if np.linalg.norm(center_point-filtered_point) < distance_threshold:
                    break
            else:
                filtered_center_points.append(center_point)

    return filtered_center_points


def track_marker_center_points(image: np.ndarray, 
                               area_threshold: int, 
                               circularity_threshold: float, 
                               centroid_threshold: int,
                               center_image_point: tuple):
    """Track center points of markers in tactile images"""

    x0, y0 = center_image_point

    # Canny edge detection, minVal = 20; maxVal = 200
    edges = cv2.Canny(image, 30, 150)

    # Find contours of the image edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_contours = []
    target_centroids = []
    target_areas = []

    for _, contour in enumerate(contours):
        if len(np.squeeze(contour)) > 4:
            area = cv2.contourArea(contour)
            good_area = area > area_threshold
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / perimeter ** 2
            good_circularity = circularity > circularity_threshold
            if good_circularity and good_area:
                local_mnt = cv2.moments(contour)
                centroid = np.array([int(local_mnt['m10'] / local_mnt['m00']), int(local_mnt['m01'] / local_mnt['m00'])])
                good_centroid = np.sqrt((centroid[0]-x0)**2 + (centroid[1]-y0)**2) > centroid_threshold
                if good_centroid:
                    target_contours.append(contour)
                    target_centroids.append(centroid)
                    target_areas.append(area)

    filtered_target_centroids = remove_duplicate_points(target_centroids, distance_threshold = 2)

    return (filtered_target_centroids, target_contours)

def extract_image_center_point(image: np.ndarray, 
                               area_threshold: int, 
                               circularity_threshold: float):
    """Track center points of markers in tactile images"""

    # Canny edge detection, minVal = 20; maxVal = 200
    edges = cv2.Canny(image, 10, 200)

    # Find contours of the image edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    target_contours = []
    target_centroids = []

    for _, contour in enumerate(contours):
        if len(np.squeeze(contour)) > 4:
            area = cv2.contourArea(contour)
            good_area = area > area_threshold
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / perimeter ** 2
            good_circularity = circularity > circularity_threshold
            if good_circularity and good_area:
                local_mnt = cv2.moments(contour)
                centroid = np.array([int(local_mnt['m10'] / local_mnt['m00']), int(local_mnt['m01'] / local_mnt['m00'])])
                target_contours.append(contour)
                target_centroids.append(centroid)

    filtered_target_centroids = remove_duplicate_points(target_centroids, distance_threshold = 2)

    return (filtered_target_centroids, target_contours)

def crop_image_at_center(image: np.ndarray, dsize: tuple) -> np.ndarray:
    """Crop at the center of the image"""
    center = image.shape
    height = dsize[0]
    width = dsize[1]
    x_dist = center[1] / 2 - width / 2
    y_dist = center[0] / 2 - height / 2

    return image[
        int(y_dist) : int(y_dist + height),  # noqa: E203
        int(x_dist) : int(x_dist + width),  # noqa: E203
    ]


def extract_image_region_with_color(image: np.ndarray, 
                                    color: np.ndarray, 
                                    color_tolerance: int) -> np.ndarray:
    """extract the region of interested by specified color"""

    lower_bound = np.clip(color - color_tolerance, 0, 255).astype(np.uint8)
    upper_bound = np.clip(color + color_tolerance, 0, 255).astype(np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)  # the black mask

    return mask

def blur_image(image: np.ndarray, filter_size: int) -> np.ndarray:
    """Blur input image with specified filter size"""
    return cv2.GaussianBlur(image, (filter_size, filter_size), 0)


def rbg_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RBG image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_binary_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Generate binary image from grayscale by thresholding"""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to desired size through interpolation"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def imshow(image):
    """Make possible to show image in jupyter"""
    # image = img_as_ubyte(image)
    _, ret = cv2.imencode(".jpg", image)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def img_float_to_ubyte(image):
    """Adjust image to make integer if it is in float format"""
    return img_as_ubyte(image)


def apply_mask_to_image(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply circular mask at center of image"""
    sizes = image.shape
    center = (int(sizes[1] / 2), int(sizes[1] / 2))
    rows, columns = np.ogrid[: sizes[0], : sizes[1]]
    dist_from_center = np.sqrt((columns - center[0]) ** 2 + (rows - center[1]) ** 2)

    mask = dist_from_center <= radius
    if len(sizes) == 2:
        mask.astype(int)
    elif len(sizes) == 3:
        mask = np.repeat(mask.astype(int).reshape(sizes[0], sizes[1], 1), 3, axis = 2)
        
    return np.multiply(image, mask)


def apply_invert_mask_to_image(image: np.ndarray, radius: int, center: tuple) -> np.ndarray:
    """Apply circular mask at center of image"""
    sizes = image.shape
    rows, columns = np.ogrid[: sizes[0], : sizes[1]]
    dist_from_center = np.sqrt((columns - center[0]) ** 2 + (rows - center[1]) ** 2)

    mask = dist_from_center >= radius
    
    if len(sizes) == 2:
        mask.astype(int)
    elif len(sizes) == 3:
        mask = np.repeat(mask.astype(int).reshape(sizes[0], sizes[1], 1), 3, axis = 2)

    return np.multiply(image, mask)


def img2tensor(img: np.ndarray) -> torch.Tensor:
    """Convert image to tensor"""
    img = (img / 255.0).astype(np.float32)
    return torch.from_numpy(img)


def tensor2img(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to image"""
    image_tensor = tensor.data
    img_numpy = image_tensor.squeeze().cpu().float().numpy()
    img_numpy = img_numpy * 255.0
    return img_numpy.astype(np.uint8)