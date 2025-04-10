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

def create_bilinear_interpolation_function(img: np.ndarray)-> np.ufunc:
    
    def bilinear_interpolation_function(value: int, indexes: tuple) -> int:
        x,y = indexes
        A = np.zeros(shape=(4,))
        neighbor_sum = [[1,0],[-1,0],[0,1],[0,-1]]

        pixel_neighbors = [[x,y],[x,y],[x,y],[x,y]]

        for index, value in enumerate(neighbor_sum):
            pixel_neighbors[
            
            
        


        interpolated_value = 
        return interpolated_value
    return np.frompyfunc(bilinear_interpolation_function, 2, 1)


def translate_image(img: np.ndarray, shift_x: int = 1, shift_y: int = 1) -> np.ndarray:
    affine_matrix = [[shift_x, 0, 0], [0, shift_y, 0], [0, 0, 1]]

    trans_img = np.zeros(shape=(img.shape), dtype=img.dtype)

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix

        if 0 <= i_out <= img.shape[0] - 1 or 0 <= j_out <= img.shape[1] - 1:
            trans_img[i_out, j_out] = value
        else:
            pass

    return img


def rotate_image_90_degrees(img: np.ndarray, rotations=1) -> np.ndarray:
    cos = np.cos((np.pi / 2) * rotations)
    sin = np.sin((np.pi / 2) * rotations)

    affine_matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    img_rotated = np.zeros(shape=img.shape, dtype=img.dtype)

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        if 0 <= i_out <= img.shape[0] - 1 or 0 <= j_out <= img.shape[1] - 1:
            img_rotated[i_out, j_out] = value
        else:
            pass

    return img_rotated


def stretch_image(
    img: np.ndarray, scalar_x: float = 1.5, scalar_y: float = 1.0
) -> np.ndarray:
    img_stretched = np.zeros(shape=img.shape, dtype=np.uint8)

    affine_matrix = [[scalar_x, 0, 0], [0, scalar_y, 0], [0, 0, 1]]

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.ndarray([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        if 0 <= i_out <= img.shape[0] - 1 or 0 <= j_out <= img.shape[1] - 1:
            img_stretched[i_out, j_out] = value
        else:
            pass

    return img_stretched


def mirror_image_horizontally(img: np.ndarray) -> np.ndarray:
    mirror_img = np.empty(shape=img.shape, dtype=np.uint8)

    affine_matrix = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for (i, j), value in np.ndenumerate(img):
        input_coords = np.ndarray([i, j, 1])
        i_out, j_out, _ = input_coords @ affine_matrix
        if 0 <= i_out <= img.shape[0] - 1 or 0 <= j_out <= img.shape[1] - 1:
            mirror_img[i_out, j_out] = value
        else:
            pass

    return mirror_img


def barrel_distort_image(img: np.ndarray) -> np.ndarray:
    return


def apply_geometric_transformations(img: np.ndarray) -> dict:
    translated_img = translate_image(img)
    mirrored_img = mirror_image_horizontally(img)
    distorted_img = barrel_distort_image(img)
    rotated_img = rotate_image_90_degrees(img)
    stretched_img = stretch_image(img)

    return {
        "translated": translated_img,
        "rotated": rotated_img,
        "stretched": stretched_img,
        "mirrored": mirrored_img,
        "distorted": distorted_img,
    }
