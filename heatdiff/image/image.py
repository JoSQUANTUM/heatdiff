"""Image loading and preprocessing utilities."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_image(
    path: str, size: Tuple[int, int] = (256, 256), grayscale: bool = True
) -> NDArray:
    """Load and preprocess an image.

    Args:
        path: Path to image file
        size: Target size for resizing (width, height)
        grayscale: Whether to convert to grayscale

    Returns:
        Image as numpy array (2D for grayscale, 3D for color)
    """
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    img = img.resize(size, resample=Image.LANCZOS)
    return np.array(img)


def add_noise(image: NDArray, mean: float = 0, std: float = 25) -> NDArray:
    """Add Gaussian noise to an image.

    Args:
        image: Input image array
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise

    Returns:
        Noisy image array
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def normalize_image(image: NDArray) -> NDArray:
    """Normalize image to [0, 1] range.

    Args:
        image: Input image array

    Returns:
        Image normalized to [0, 1] range
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def normal_range(image: NDArray) -> NDArray:
    """Convert image to [0, 255] uint8 range.

    Args:
        image: Input image array

    Returns:
        Image scaled to [0, 255] as uint8
    """
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (normalized * 255).astype(np.uint8)
