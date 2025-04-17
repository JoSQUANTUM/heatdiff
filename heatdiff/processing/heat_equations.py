"""Forward, backward and reverse heat equations for images.

This module implements various heat equation operations on images,
complementing the segmentation functionality in the package.
"""

from typing import List, Tuple

from numpy.typing import NDArray
from scipy import signal


def bwd_heat_equation(
    kernel: NDArray,
    image: NDArray,
    m: int,
) -> List[NDArray]:
    """Backwards heat equation for images.

    Args:
        kernel: Heat kernel approximation
        image: Original image as numpy array
        m: Number of convolutions to perform

    Returns:
        List of images corresponding to the backward
        heat equation at each step
    """
    result_list = [image]
    for i in range(1, m + 1):
        current_image = result_list[i - 1]
        convolved_image = signal.fftconvolve(current_image, kernel, mode="same")
        up = current_image - convolved_image
        result_list.append(up)
    return result_list


def heat_equation(kernel: NDArray, image: NDArray, m: int) -> List[NDArray]:
    """Forwards heat equation for images.

    Args:
        kernel: Heat kernel approximation
        image: Original image as numpy array
        m: Number of convolutions to perform

    Returns:
        List of images corresponding to the
        heat equation at each step
    """
    result_list = [image]
    for i in range(1, m + 1):
        current_image = result_list[i - 1]
        convolved_image = signal.fftconvolve(
            current_image,
            kernel,
            mode="same",
        )
        up = convolved_image - current_image
        result_list.append(up)
    return result_list


def reverse_heat_equation(
    kernel: NDArray, image: NDArray, m: int
) -> Tuple[List[NDArray], List[NDArray]]:
    """Reverse heat equation; an analogue of the reverse Wiener process.

    Args:
        kernel: Heat kernel approximation
        image: Original image as numpy array
        m: Number of convolutions to perform

    Returns:
        Tuple containing:
        - List of heat equation results
        - List of convolution results
    """
    result_list = [image]
    heat = [None] * (m + 1)
    for _ in range(1, m):
        current_image = result_list[-1]
        result_list.append(
            signal.fftconvolve(
                current_image,
                kernel,
                mode="same",
            )
        )

    for i in range(m + 1):
        heat[m - i - 1] = result_list[m - i - 1] - result_list[m - 1]
    return heat, result_list


def heat_semigroup(kernel: NDArray, image: NDArray, m: int) -> List[NDArray]:
    """Convolution of the heat kernel with images.

    Args:
        kernel: Heat kernel approximation
        image: Original image as numpy array
        m: Number of convolutions to perform

    Returns:
        List of images corresponding to the
        convolution of the heat kernel at each step
    """
    result_list = [image]
    for i in range(1, m + 1):
        current_image = result_list[i - 1]
        convolved_image = signal.fftconvolve(
            current_image,
            kernel,
            mode="same",
        )
        result_list.append(convolved_image)
    return result_list
