import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from typing import Any, NewType


image_rgb = NewType("image_rgb", np.ndarray)
image = NewType("image", np.typing.NDArray)
image_name = NewType("image_name", str)

# leitura de imagens
img = cv.imread("../img/lena.png")  # Le imagem em BGR
img[:, :, 0]  # retorna o canal 0 da imagem

# Visualizacao

# subplots
fig, a = plt.subplots(2, 2, figsize=(15, 15))
a[0, 0].imshow(img)
a[0, 0].set_title("titulo")

# Iterar em cada canal da imagem
for channel in range(0, img.shape[2]):
    curr_channel = img[:, :, channel]

    for i, v in np.ndenumerate(curr_channel):
        print(i, v)

# numpy

# Colocar imagens lado a lado
np.hstack((img, img))

# probabilidade
hist, _ = np.histogram(img, bins=256)
hist.cumsum()  # array de somas cumulativas


# Criacao de arrays
np.zeros(shape=(3, 3), dtype=np.integer)
np.ones(shape=(3, 3), dtype=np.integer)
np.empty(shape=(3, 3), dtype=np.integer)


# Opencv

# Processamento de histograma
# Equalizacao de histograma
cv.equalizeHist(img)

# histogram matching
ref = img
match_histograms(img, ref, multichannel=True)

# Transformada de fourier(Dominio Frequencia)
cv.dft(img)
cv.idft(img)


# Filtros
# Filtros de deteccao de borda
cv.Laplacian(img, cv.CV_16S)
cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
