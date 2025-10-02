# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import cv2 as cv


def translate_image(img: np.ndarray, shift_x: int = 1, shift_y: int = 1) -> np.ndarray:
    affine_matrix = [[shift_x, 0, 0], [0, shift_y, 0], [0, 0, 1]]
    trans_img = np.zeros(shape=(img.shape), dtype=img.dtype)
    for (i, j), value in np.ndenumerate(img):
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        trans_img[i_out, j_out] = value

    return img


def rotate_image_90_degrees(img: np.ndarray, rotations=1) -> np.ndarray:
    cos = np.cos((np.pi / 2) * rotations)
    sin = np.sin((np.pi / 2) * rotations)

    affine_matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    img_rotated = np.zeros(shape=img.shape, dtype=img.dtype)

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        img_rotated[i_out, j_out] = value

    return img_rotated


def stretch_image(
    img: np.ndarray, scalar_x: float = 1.5, scalar_y: float = 1.0
) -> np.ndarray:
    img_stretched = np.empty(
        shape=(np.uint16(img.shape[0] * scalar_x), np.uint16(img.shape[1] * scalar_y)),
        dtype=np.uint8,
    )

    affine_matrix = [[scalar_x, 0, 0], [0, scalar_y, 0], [0, 0, 1]]

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.ndarray([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        img_stretched[i_out, j_out] = value

    return img_stretched


def mirror_image_horizontally(img: np.ndarray) -> np.ndarray:
    return


def barrel_distort_image(img: np.ndarray) -> np.ndarray:
    return


def apply_geometric_transformations(img: np.ndarray) -> dict:
    trans_img = translate_image(img, 1, 1)
    mirror_img = mirror_image_horizontally(img)
    dist_img = barrel_distort_image(img)
    rot_img = rotate_image_90_degrees(img)
    str_img = stretch_image(img)

    return {
        "translated": trans_img,
        "rotated": rot_img,
        "stretched": str_img,
        "mirrored": mirror_img,
        "distorted": dist_img,
    }
