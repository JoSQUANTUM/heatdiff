"""Visualization utilities for segmentation results."""

import matplotlib.pyplot as plt
from numpy.typing import NDArray


def show_image(image: NDArray, title: str = "", cmap: str = "gray") -> None:
    """Display an image with optional title.

    Args:
        image: Image data as numpy array
        title: Title for the plot
        cmap: Colormap to use
    """
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=20)
    plt.show()
