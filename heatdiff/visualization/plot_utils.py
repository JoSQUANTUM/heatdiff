"""Advanced plotting utilities for heat diffusion analysis."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_heat_kernels(
    heat_kernel: callable, heat_approx: callable, t: float, n: int
) -> None:
    """Plot heat kernel against its approximation.

    Args:
        heat_kernel: Function that computes Euclidean heat kernel
        heat_approx: Function that computes approximate heat kernel
        t: Time step size
        n: Size of the kernel
    """
    heat = heat_kernel(t, n)
    approx = heat_approx(t, n)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(heat, cmap="hot", interpolation="nearest")
    ax1.set(xlabel="X", ylabel="Y", title="Heat Kernel")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(approx, cmap="hot", interpolation="nearest")
    ax2.set(xlabel="X", ylabel="Y", title="Approximation")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()


def plot_3d(
    function: callable,
    *args,
    x_range: Tuple[float, float] = (-1, 1),
    y_range: Tuple[float, float] = (-1, 1),
    resolution: int = 256
) -> None:
    """Plot 2D functions in 3D space.

    Args:
        function: 2D function to plot
        *args: Additional arguments for the function
        x_range: X-axis range
        y_range: Y-axis range
        resolution: Plot resolution
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y, *args)

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Function Plot")


def compare_3d_images(image1: NDArray, image2: NDArray) -> None:
    """Compare two images as 3D surface plots.

    Args:
        image1: First image array
        image2: Second image array
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
    )

    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, image1, cmap="viridis", edgecolor="none")
    ax2.plot_surface(X, Y, image2, cmap="viridis", edgecolor="none")

    ax1.set(xlabel="X", ylabel="Y", zlabel="Z", title="Image 1")
    ax2.set(xlabel="X", ylabel="Y", zlabel="Z", title="Image 2")

    plt.tight_layout()


def compare_norm_histograms(image1: NDArray, image2: NDArray) -> None:
    """Plot normalized histograms of two images for comparison.

    Args:
        image1: First image array
        image2: Second image array
    """
    plt.hist(image1.ravel(), bins=256, color="blue", alpha=0.7, label="Image 1")
    plt.hist(image2.ravel(), bins=256, color="red", alpha=0.7, label="Image 2")
    plt.title("Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
