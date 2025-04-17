"""Image comparison and accuracy metrics."""

import numpy as np
from numpy.typing import NDArray


def mean_squared_error(image1: NDArray, image2: NDArray) -> float:
    """Calculate Mean Squared Error between two images.

    Args:
        image1: First input image
        image2: Second input image to compare

    Returns:
        MSE value normalized by number of pixels

    Raises:
        ValueError: If images have different dimensions
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")

    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    return err / float(image1.size)
