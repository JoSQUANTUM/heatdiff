"""Image compression utilities using heat diffusion methods."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from heatdiff.math.weighted_kmeans import HermiteWeightedKMeans, JacobiWeightedKMeans


def jacobi_compression(
    image: NDArray,
    num_clusters: int,
    t: float = 2,
    tolerance: float = 0.001,
    max_iter: int = 500,
) -> Tuple[NDArray, NDArray]:
    """Compress image using Jacobi-weighted K-Means clustering.

    Args:
        image: Input grayscale image (2D array)
        num_clusters: Number of color clusters
        t: Time parameter for Jacobi K-Means
        tolerance: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Tuple containing:
        - encoded_image: Compressed image (uint8)
        - centroids: Cluster centroids

    Raises:
        ValueError: If input image is empty
    """
    h, w = image.shape
    image_flat = image.flatten().reshape(-1, 1)

    if image_flat.size == 0:
        raise ValueError("Cannot cluster an empty image")

    kmeans = JacobiWeightedKMeans(
        k=num_clusters, t=t, tolerance=tolerance, max_iter=max_iter
    )
    kmeans.fit(image_flat)
    labels = kmeans.predict(image_flat)
    centroids = kmeans.centroids

    return centroids[labels].reshape(h, w).astype(np.uint8), centroids


def hermite_compression(
    image: NDArray,
    num_clusters: int,
    n: int = 0,
    tolerance: float = 0.001,
    max_iter: int = 500,
) -> Tuple[NDArray, NDArray]:
    """Compress image using Hermite-weighted K-Means clustering.

    Args:
        image: Input grayscale image (2D array)
        num_clusters: Number of color clusters
        n: Order for Hermite K-Means
        tolerance: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Tuple containing:
        - encoded_image: Compressed image (uint8)
        - centroids: Cluster centroids

    Raises:
        ValueError: If input image is empty
    """
    h, w = image.shape
    image_flat = image.flatten().reshape(-1, 1)

    if image_flat.size == 0:
        raise ValueError("Cannot cluster an empty image")

    kmeans = HermiteWeightedKMeans(
        k=num_clusters, n=n, tolerance=tolerance, max_iter=max_iter
    )
    kmeans.fit(image_flat)
    labels = kmeans.predict(image_flat)
    centroids = kmeans.centroids

    return centroids[labels].reshape(h, w).astype(np.uint8), centroids
