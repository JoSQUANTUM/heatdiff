"""Forward, backward and reverse heat equations for images"""

from scipy import signal


def bwd_heat_equation(kernel, image, m):
    """
     Backwards heat equation for images

     Parameters:
    - kernel: Heat kernel approximation
    - image: Original image.
    - m: Number of convolutions to perform

    Returns:
    - result_list: List of images corresponding to the backward heat equation at
    each convolution step
    """
    result_list = [image]
    for i in range(1, m + 1):
        convolved_image = signal.convolve2d(result_list[i - 1], kernel, mode="same")
        up = result_list[i - 1] - convolved_image
        result_list.append(up)
    return result_list


def heat_equation(kernel, image, m):
    """
     Forwards heat equation for images.

     Parameters:
    - kernel: Heat kernel approximation
    - image: Original image.
    - m: Number of convolutions to perform

    Returns:
    - result_list: List of images corresponding to the heat equation at
    each convolution step
    """
    result_list = [image]
    for i in range(1, m + 1):
        convolved_image = signal.convolve2d(result_list[i - 1], kernel, mode="same")
        up = convolved_image - result_list[i - 1]
        result_list.append(up)
    return result_list


def reverse_heat_equation(kernel, image, m):
    """
     Reverse heat equation; an analogue of the reverse Wiener process.

     Parameters:
    - kernel: Heat kernel approximation
    - image: Original image.
    - m: Number of convolutions to perform

    Returns:
    - result_list: List of images corresponding to the reverse heat equation at
    each convolution step
    """
    result_list = [image]
    heat = [None] * (m + 1)
    for i in range(1, m):
        result_list.append(signal.convolve2d(result_list[-1], kernel, mode="same"))

    for i in range(m + 1):
        heat[m - i - 1] = result_list[m - i - 1] - result_list[m - 1]
    return heat, result_list


def heat_semigroup(kernel, image, m):
    """
     Convolution of the heat kernel approximation with images.

     Parameters:
    - kernel: Heat kernel approximation
    - image: Original image.
    - m: Number of convolutions to perform

    Returns:
    - result_list: List of images corresponding to the convolution of the heat kernel at
    each convolution step
    """
    result_list = [image]
    for i in range(1, m + 1):
        convolved_image = signal.convolve2d(result_list[i - 1], kernel, mode="same")
        result_list.append(convolved_image)
    return result_list
