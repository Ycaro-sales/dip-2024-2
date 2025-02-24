import numpy as np
import math


def get_image_info(image: np.ndarray):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """

    width = image.shape[0]
    height = image.shape[1]
    depth = np.ma.minimum_fill_value(image) + 1
    dtype = type(image.dtype)
    max_val = image.max()
    min_val = image.min()
    mean_val = np.mean(image)
    std_val = np.std(image)

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }


# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")
