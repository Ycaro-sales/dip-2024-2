import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def display_color_histograms(img: np.ndarray):
    red = img[:, :, 2]
    green = img[:, :, 1]
    blue = img[:, :, 0]

    _, axs = plt.subplots(3)

    axs[0].hist(red.ravel(), bins=256)
    axs[1].hist(green.ravel(), bins=256)
    axs[2].hist(blue.ravel(), bins=256)

    plt.show()


# if __name__ == "__main__":
#     main()
