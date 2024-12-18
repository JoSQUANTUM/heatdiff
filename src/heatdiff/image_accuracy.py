"""The mse function is adapted from: https://pyimagesearch.com/2014/09/15/python-compare-two-images/"""

import numpy as np


def mse(img1, img2):
    """
    Calculate the Mean Squared Error (MSE) between two images.

    This function computes the MSE, which is a measure of the average squared difference
    between the pixel values of two images. A lower MSE indicates greater similarity
    between the images.

    Parameters:
    -----------
    img1 : numpy.ndarray
        The first input image.
    img2 : numpy.ndarray
        The second input image to compare with the first.

    Returns:
    --------
    float
        The Mean Squared Error between the two input images.

    Notes:
    ------
    - The two input images must have the same dimensions.
    - The function converts the input images to float type for calculation.
    - The MSE is normalized by the total number of pixels in the image.

    Raises:
    -------
    ValueError
        If the input images have different shapes.
    """
    # Check if the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Calculate the squared difference between the two images
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)

    # Normalize by the number of pixels
    err /= float(img1.shape[0] * img1.shape[1])

    return err
