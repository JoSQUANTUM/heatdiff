"""Utility functions for preprocessing; convolution, resizing etc", k_means. 
Need to refactor to reduce duplicacy with the 3 k-means methods."""

import numpy as np


def normalize_image(image):
    """
    Normalize the input image to the range [0, 1].

    This function applies min-max normalization to the input image,
    scaling pixel values to be between 0 and 1.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be normalized.

    Returns:
    --------
    numpy.ndarray
        The normalized image with pixel values in the range [0, 1].

    """
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image


def normal_range(image):
    """
    Return the input image to the range [0, 255] and convert to 8-bit unsigned integer.

    This function applies min-max normalization to the input image,
    scaling pixel values to be between 0 and 255, and then converts
    the result to an 8-bit unsigned integer format.

    Parameters:
    -----------
    image : numpy.ndarray
        The input image to be normalized.

    Returns:
    --------
    numpy.ndarray
        The normalized image with pixel values in the range [0, 255]
        as 8-bit unsigned integers (uint8).
    """
    return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
