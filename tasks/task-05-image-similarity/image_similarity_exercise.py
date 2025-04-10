# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np


def mean_squared_error(i1: np.ndarray, i2: np.ndarray) -> float:
    squared_errors = (i1 - i2) ** 2
    mean = np.mean(squared_errors)

    return float(mean)


def peak_signal_to_noise_ratio(i1: np.ndarray, i2: np.ndarray) -> float:
    log10 = np.emath.log10
    max_i1 = np.max(i1)
    mse = mean_squared_error(i1, i2)

    psnr = 20 * log10(max_i1) - 10 * log10(mse)

    return psnr


def structural_similarity_index(
    i1: np.ndarray, i2: np.ndarray, k1: float = 0.01, k2: float = 0.03, L=1
) -> float:
    mean_i1 = np.mean(i1)
    mean_i2 = np.mean(i2)

    sm_i1 = mean_i1**2
    sm_i2 = mean_i2**2

    var_i1 = np.var(i1)
    var_i2 = np.var(i2)

    covar_i1_i2 = np.cov(i1.flatten(), i2.flatten())[0][1]

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    numerator = ((2 * mean_i1 * mean_i2) + c1) * ((2 * covar_i1_i2) + c2)
    denominator = (sm_i1 + sm_i2 + c1) * (var_i1 + var_i2 + c2)

    ssim = numerator / denominator

    return ssim


def normalized_pearson_correlation_coefficient(i1: np.ndarray, i2: np.ndarray) -> float:
    covar_i1_i2 = np.cov(i1.flatten(), i2.flatten())[0][1]
    std_i1 = np.std(i1)
    std_i2 = np.std(i2)
    pcc = covar_i1_i2 / (std_i1 * std_i2)

    npcc = (1 / 255) * pcc

    return npcc


def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    mse: float = mean_squared_error(i1, i2)
    psnr: float = peak_signal_to_noise_ratio(i1, i2)
    ssim: float = structural_similarity_index(i1, i2)
    npcc: float = normalized_pearson_correlation_coefficient(i1, i2)

    return {"mse": mse, "psnr": psnr, "ssim": ssim, "npcc": npcc}
