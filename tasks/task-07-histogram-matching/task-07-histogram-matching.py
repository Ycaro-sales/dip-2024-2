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
import matplotlib.pyplot as plt


RESOLUTION_8_BIT_DEPTH = 2**8


def create_histogram_equalizer_function(img: np.ndarray) -> Callable:
    img_histogram, _ = np.histogram(img, bins=256)
    total_px = img.size

    def equalizer(r: int) -> np.uint8:
        cumulative_sum_r = img_histogram.cumsum()[r]
        return np.uint8(np.rint((cumulative_sum_r / total_px) * RESOLUTION_8_BIT_DEPTH))

    return equalizer


def histogram_equalization(img: np.ndarray):
    equalizer = create_histogram_equalizer_function(img)
    equalized_img = np.apply_over_axes(equalizer, img, img.shape)

    return equalized_img


def create_inverse_histogram_equalizer_function(img: np.ndarray) -> np.vectorize:
    img_hist_equalize_function = create_histogram_equalizer_function(img)

    mapping: dict[int, int] = dict()
    equalized_values = img_hist_equalize_function(range(0, RESOLUTION_8_BIT_DEPTH))
    for index, value in enumerate(equalized_values):
        mapping[value] = index

    def inverse_histogram_equalizer(s: int) -> int:
        return mapping[s]

    return np.vectorize(inverse_histogram_equalizer)


# OK
def match_histograms(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_histogram_image = np.zeros(shape=source_img.shape)

    histogram_equalized_src_img = histogram_equalization(source_img)

    inverse_histogram_equalizer_function = create_inverse_histogram_equalizer_function(
        reference_img
    )

    for (i, j), value in np.ndenumerate(histogram_equalized_src_img):
        matched_histogram_image[i, j] = inverse_histogram_equalizer_function(value)

    return matched_histogram_image


# OK
def rgb_histogram_equalization(img: np.ndarray) -> np.ndarray:
    img_red = img[SLICE_RED_RGB_IMAGE]
    img_green = img[SLICE_GREEN_RGB_IMAGE]
    img_blue = img[SLICE_BLUE_RGB_IMAGE]

    equalized_red = histogram_equalization(img_red)
    equalized_green = histogram_equalization(img_green)
    equalized_blue = histogram_equalization(img_blue)

    equalized_rgb_img = np.zeros(shape=img.shape, dtype=img.dtype)

    equalized_rgb_img[SLICE_RED_RGB_IMAGE] = equalized_red
    equalized_rgb_img[SLICE_GREEN_RGB_IMAGE] = equalized_green
    equalized_rgb_img[SLICE_BLUE_RGB_IMAGE] = equalized_blue

    return equalized_rgb_img


def match_histograms_rgb(
    source_img: np.ndarray, reference_img: np.ndarray
) -> np.ndarray:


def main():
    source_img = cv.imread("./source.jpg")
    reference_img = cv.imread("./reference.jpg")

    print(source_img.shape)
    print(reference_img.shape)

    histogram_r, histogram_g, histogram_b = create_rgb_image_histograms(source_img)

    equalized_source_img = rgb_histogram_equalization(source_img)

    e_r, e_g, e_b = create_rgb_image_histograms(equalized_source_img)

    cv.imshow("source rgb", source_img)
    cv.imshow("equalized source rgb", equalized_source_img)

    print(histogram_r.shape)

    # fig, axs = plt.subplots(2, 3)
    # axs[0, 0].bar(range(0, 256), histogram_r)
    # axs[0, 1].bar(range(0, 256), histogram_g)
    # axs[0, 2].bar(range(0, 256), histogram_b)
    #
    # axs[1, 0].bar(range(0, 256), e_r)
    # axs[1, 1].bar(range(0, 256), e_g)
    # axs[1, 2].bar(range(0, 256), e_b)

    cv.waitKey(0)
    # plt.show()


if __name__ == "__main__":
    main()
