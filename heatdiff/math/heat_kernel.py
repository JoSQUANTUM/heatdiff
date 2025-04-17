"""Heat kernel approximations for image processing.

This module provides various implementations of heat kernels and their
approximations for use in diffusion-based image processing.
"""

import numpy as np
from scipy import special


def heat_kernel_1d(t: float, n: int) -> np.ndarray:
    """Continuous 1D heat kernel on Euclidean space.

    Args:
        t: Time step size
        n: Kernel size

    Returns:
        Normalized 1D convolution kernel
    """
    x = np.linspace(-(n - 1) / 2.0, n / 2)
    kernel = np.exp(-0.5 * (np.square(x)) / t)
    return kernel / np.sum(kernel)


def heat_kernel_2d(t: float, n: int) -> np.ndarray:
    """Continuous 2D heat kernel.

    Args:
        t: Time step size
        n: Kernel size

    Returns:
        Normalized 2D convolution kernel
    """
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)
    coeff = (2 * np.pi * t) ** -1
    exponent = -0.5 * (np.square(xx) + np.square(yy)) / t
    kernel = coeff * np.exp(exponent)
    return kernel / np.sum(kernel)


def jacobi_theta_2d(t: float, m: int) -> np.ndarray:
    """2D Jacobi-Theta approximation of heat kernel on square.

    Args:
        t: Time step size
        m: Kernel size

    Returns:
        Normalized 2D convolution kernel
    """

    def infinite_sum(n, s):
        return (
            np.exp(-(n**2 + s**2) * np.pi * t)
            * np.cos(2 * np.pi * n * x)
            * np.cos(2 * np.pi * s * y)
        )

    x, y = np.meshgrid(
        np.linspace(-(m - 1) / 2.0, m / 2, m), np.linspace(-(m - 1) / 2.0, m / 2, m)
    )
    f = 1 + 2 * np.sum(
        [infinite_sum(n, s) for n in range(1, 100) for s in range(1, 100)], axis=0
    )
    return f / np.sum(f)


def bessel_kernel_2d(t: float, n: int) -> np.ndarray:
    """2D Bessel approximation of heat kernel on disk.

    Args:
        t: Time step size
        n: Kernel size

    Returns:
        Normalized 2D convolution kernel
    """
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)
    kernel = special.iv(np.square(xx) + np.square(yy), t)
    return kernel / np.sum(kernel)
