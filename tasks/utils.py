import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import NewType

image_rgb = NewType("image_rgb", np.ndarray)
image = NewType("image", np.ndarray)
image_name = NewType("image_name", str)

BGR_RED_SLICE = np.s_[:, :, 2]
BGR_GREEN_SLICE = np.s_[:, :, 1]
BGR_BLUE_SLICE = np.s_[:, :, 0]

red_slice = np.s_[:, :, 0]
green_slice = np.s_[:, :, 1]
blue_slice = np.s_[:, :, 2]

imgs = {}

imgs["baboon"] = cv.imread("../../img/baboon.png")
imgs["chips"] = cv.imread("../../img/chips.png")
imgs["lena"] = cv.imread("../../img/lena.png")
imgs["rgb"] = cv.imread("../../img/rgb.png")
imgs["rgbcube_kBKG"] = cv.imread("../../img/rgbcube_kBKG.png")
imgs["flowers"] = cv.imread("../../img/flowers.jpg")
imgs["hsv_disk"] = cv.imread("../../img/hsv_disk.png")
imgs["monkey"] = cv.imread("../../img/monkey.jpeg")
imgs["strawberries"] = cv.imread("../../img/strawberries.tif")


def plot_color_channels(img: image, name: image_name):
    im_red = img[red_slice]
    im_green = img[green_slice]
    im_blue = img[blue_slice]

    _, a = plt.subplots(1, 3, figsize=(15, 15))
    a[0].imshow(im_red, cmap="gray")
    a[0].set_title(f"{name} red")
    a[1].imshow(im_green, cmap="gray")
    a[1].set_title(f"{name} green")
    a[2].imshow(im_blue, cmap="gray")
    a[2].set_title(f"{name} blue")

    plt.show()


def display_histograms(img: np.ndarray, im_name: str):
      red_hist, _ = np.histogram(img[red_slice], bins=256)
      green_hist, _ = np.histogram(img[green_slice], bins=256)
      blue_hist, _ = np.histogram(img[blue_slice], bins=256)

      fig, axs = plt.subplots(1,3, figsize=(15,6))

      axs[0].bar(range(0,256), red_hist, width=1, color = "red")
      axs[0].set_title(f'{im_name} Red')

      axs[1].bar(range(0,256), green_hist, width=1, color = "green")
      axs[1].set_title(f'{im_name} Green')

      axs[2].bar(range(0,256), blue_hist, width=1, color = "blue")
      axs[2].set_title(f'{im_name} Blue')

      plt.show()

def laplace_filter(img: image):
    ddepth = cv.CV_16S

    red = img[red_slice]
    green = img[green_slice]
    blue = img[blue_slice]

    sobel_img = np.zeros(shape=img.shape, dtype=img.dtype)
    lap_img = np.zeros(shape=img.shape, dtype=img.dtype)

    sobel_img[red_slice] = cv.Sobel(
        red,
        ddepth,
        1,
        0,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv.BORDER_DEFAULT,
    )

    sobel_img[green_slice] = cv.Sobel(
        green,
        ddepth,
        1,
        0,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv.BORDER_DEFAULT,
    )

    sobel_img[blue_slice] = cv.Sobel(
        blue,
        ddepth,
        1,
        0,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv.BORDER_DEFAULT,
    )

    lap_img[red_slice] = cv.Laplacian(red, ddepth)
    lap_img[green_slice] = cv.Laplacian(green, ddepth)
    lap_img[blue_slice] = cv.Laplacian(blue, ddepth)

def sobel_filter(img: image):


def blur_image(img: image):
    img_blur = cv.GaussianBlur(img, (5, 5), 0)
    return img_blur
