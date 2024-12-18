"""Heat kernel approximations in 1 and 2D. """

import scipy
import numpy as np


def heat_kernel_1d(t, n):
    """
    Continuous 1D heat kernel on Euclidean space

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (array): normalized convolution kernel
    """
    x = np.linspace(-(n - 1) / 2.0, n / 2)

    kernel = np.exp(-0.5 * (np.square(x)) / t)

    return kernel / np.sum(kernel)


def jacobi_theta_1d(t, n):
    """
    1D Jacobi-Theta approximation of the heat kernel on the square.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (array): normalized convolution kernel
    """
    n = np.linspace(
        0,
        n,
    )

    kernel = 1 + 2 * np.exp(-t * np.pi * n**2) * np.cos(n * np.pi * x)
    return kernel / np.sum(kernel)


def bes1(t, n):
    """
    1D Bessel approximation of the heat kernelon the disk.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (array): normalized convolution kernel
    """
    x = np.linspace(-(n - 1) / 2.0, n / 2)
    heat_kernel = scipy.special.iv(np.square(x), t)
    return heat_kernel / np.sum(heat_kernel)


def bes2(t, n):
    """
    2D Bessel approximation of the heat kernel on the disk.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (matrix): normalized convolution kernel
    """
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)

    kernel = scipy.special.iv(np.square(xx) + np.square(yy), t)

    return kernel / np.sum(kernel)


def inv_bes2(t, n):
    """
    2D Bessel approximation of the 'inverse' heat kernel on the disk.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (matrix): normalized convolution kernel
    """
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)

    kernel = scipy.special.kv(np.square(xx) + np.square(yy), t)

    return kernel / np.sum(kernel)


def heat_kernel_2d(t, n):
    """
    Continuous 2D heat kernel.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (matrix): normalized convolution kernel
    """

    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (2 * np.pi * t) ** -1 * np.exp(-0.5 * (np.square(xx) + np.square(yy)) / t)

    return kernel / np.sum(kernel)


def jacobi_theta_2d(t, m):
    """
    2D Jacobi-Theta approximation of the heat kernel on the square.

    Parameters:
    - t (float): time step size
    - n (int): kernel size

    Returns:
    - kernel (matrix): normalized convolution kernel
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
