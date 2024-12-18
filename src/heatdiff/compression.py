import numpy as np
from .weighted_kmeans import JacobiWeightedKMeans, HermiteWeightedKMeans


def jacobi_compression(image, num_clusters, t=2, tolerance=0.001, max_iter=500):
    """
    Compress an image using weighted K-Means clustering, where the
    weight is the jacobi-theta heat kernel.

    This function applies this weighted K-Means clustering to compress the input image
    by reducing it to a specified number of color clusters.

    Parameters:
    -----------
    image : numpy.ndarray
        The input grayscale image to be compressed. Should be a 2D array.
    num_clusters : int
        The number of color clusters to use for compression.
    t : float, optional
        The time parameter for Jacobi K-Means. Default is 2.
    tolerance : float, optional
        The convergence tolerance for the clustering algorithm. Default is 0.001.
    max_iter : int, optional
        The maximum number of iterations for the clustering algorithm. Default is 500.

    Returns:
    --------
    tuple
        A tuple containing two elements:
        - encoded_image : numpy.ndarray
            The compressed image, with each pixel replaced by its cluster centroid.
        - centroids : numpy.ndarray
            The centroids (representative colors) of the clusters.

    Raises:
    -------
    ValueError
        If the input image is empty.

    Notes:
    ------
    - The input image is flattened for clustering and then reshaped to the original dimensions.
    - The output image is converted to 8-bit unsigned integer format.
    """
    h, w = image.shape
    image_flat = image.flatten().reshape(-1, 1)

    if image_flat.size == 0:
        raise ValueError("Cannot cluster an empty image")

    kmeans = JacobiWeightedKMeans(
        k=num_clusters, t=t, tolerance=tolerance, max_iter=max_iter
    )
    kmeans.fit(image_flat)

    # Predict on the entire flattened image at once
    labels = kmeans.predict(image_flat)

    centroids = kmeans.centroids

    encoded_image = centroids[labels].reshape(h, w).astype(np.uint8)
    return encoded_image, centroids


def hermite_compression(image, num_clusters, n=0, tolerance=0.001, max_iter=500):
    """
    Compress an image using weighted K-Means clustering, where the weight
    is the hermite polynomial of order n.

    This function applies Hermite K-Means clustering to compress the input image
    by reducing it to a specified number of color clusters.

    Parameters:
    -----------
    image : numpy.ndarray
        The input grayscale image to be compressed. Should be a 2D array.
    num_clusters : int
        The number of color clusters to use for compression.
    n : int, optional
        The order for the Hermite K-Means. Default is 2.
    tolerance : float, optional
        The convergence tolerance for the clustering algorithm. Default is 0.001.
    max_iter : int, optional
        The maximum number of iterations for the clustering algorithm. Default is 500.

    Returns:
    --------
    tuple
        A tuple containing two elements:
        - encoded_image : numpy.ndarray
            The compressed image, with each pixel replaced by its cluster centroid.
        - centroids : numpy.ndarray
            The centroids (representative colors) of the clusters.

    Raises:
    -------
    ValueError
        If the input image is empty.

    Notes:
    ------
    - The input image is flattened for clustering and then reshaped to the original dimensions.
    - The output image is converted to 8-bit unsigned integer format.
    """
    h, w = image.shape
    image_flat = image.flatten().reshape(-1, 1)

    if image_flat.size == 0:
        raise ValueError("Cannot cluster an empty image")

    kmeans = HermiteWeightedKMeans(
        k=num_clusters, n=n, tolerance=tolerance, max_iter=max_iter
    )
    kmeans.fit(image_flat)

    # Predict on the entire flattened image at once
    labels = kmeans.predict(image_flat)

    centroids = kmeans.centroids

    encoded_image = centroids[labels].reshape(h, w).astype(np.uint8)
    return encoded_image, centroids
