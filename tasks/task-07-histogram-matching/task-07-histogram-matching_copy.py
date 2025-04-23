# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

from typing import Callable
import cv2 as cv
import numpy as np


def create_histogram_equalizer(img: np.ndarray) -> Callable:
    histogram, _ = np.histogram(img)

    def histogram_equalizer(value: int):
        return int(histogram.cumsum()[value] / img.size * 255)

    return histogram_equalizer


def histogram_equalization(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_hist, _ = np.histogram(img, bins=256)
    mapping = np.zeros(shape=(256,), dtype=np.uint32)

    for index, _ in enumerate(mapping):
        mapping[index] = int(img_hist.cumsum()[index] / img.size * 255)

    eq_img = np.zeros(shape=img.shape, dtype=img.dtype)
    for i, v in np.ndenumerate(img):
        eq_img[i] = mapping[v]

    return (eq_img, mapping)


def inverse_hist_map(img: np.ndarray) -> dict:
    eq_img, map = histogram_equalization(img)
    inv_map = {}

    for i, v in enumerate(map):
        inv_map[v] = i

    if 0 in inv_map.keys():
        last_seen = inv_map[0]
    else:
        last_seen = 0

    for i in range(0, 256):
        if i not in inv_map.keys():
            inv_map[i] = last_seen
        else:
            last_seen = inv_map[i]

    return inv_map


def histogram_matching_color(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    eq_src, _ = histogram_equalization(src)
    _, ref_map = histogram_equalization(ref)

    inv_map = inverse_hist_map(ref_map)
    matched_img = np.zeros(shape=src.shape, dtype=np.uint8)

    for i, v in np.ndenumerate(eq_src):
        matched_img[i] = inv_map[v]

    return matched_img


def histogram_matching(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_blue = src[:, :, 0]
    ref_blue = ref[:, :, 0]
    matched_blue = histogram_matching_color(src_blue, ref_blue)

    src_green = src[:, :, 1]
    ref_green = ref[:, :, 1]
    matched_green = histogram_matching_color(src_green, ref_green)

    src_red = src[:, :, 2]
    ref_red = ref[:, :, 2]
    matched_red = histogram_matching_color(src_red, ref_red)

    match_src = np.zeros(shape=src.shape, dtype=src.dtype)
    match_src[:, :, 0] = matched_blue
    match_src[:, :, 1] = matched_green
    match_src[:, :, 2] = matched_red

    return match_src


def main():
    src_img = cv.imread("./source.jpg")
    ref_img = cv.imread("./reference.jpg")

    inv_src = np.zeros(shape=src_img.shape, dtype=np.uint8)

    inv_src_b = histogram_matching(src_img[:, :, 0], ref_img[:, :, 0])
    inv_src_g = histogram_matching(src_img[:, :, 1], ref_img[:, :, 1])
    inv_src_r = histogram_matching(src_img[:, :, 2], ref_img[:, :, 2])
    inv_src[:, :, 0] = inv_src_b
    inv_src[:, :, 1] = inv_src_g
    inv_src[:, :, 2] = inv_src_r
    inv_src_2 = histogram_matching(src_img, ref_img)

    inv_src_rgb = cv.cvtColor(inv_src_2, code=cv.COLOR_BGR2RGB)

    cv.imshow("source", src_img)
    cv.imshow("match source", inv_src_2)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
