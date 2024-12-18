"""Some plotting functions"""

import matplotlib.pyplot as plt
import numpy as np


def plot_heat_kernels(heat_kernel, heat_approx, t, n):
    """
    Plotting the heat kernel against the chosen approximation.

    Parameters:
    - heat_kernel (matrix): Euclidean heat kernel
    - heat_approx (matrix): approximate heat kernel
    - t (float): time step size
    - n (int): size of the kernel

    Returns:
    - None: This function displays a plot and does not return any value.
    """
    heat = heat_kernel(t, n)
    approx = heat_approx(t, n)

    plt.subplot(1, 2, 1)  # row 1, column 2, count 1
    plt.imshow(heat, cmap="hot", interpolation="nearest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("heat kernel")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(approx, cmap="hot", interpolation="nearest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("approximation")
    plt.colorbar()

    plt.tight_layout()


def plot_3d(function, *args, x_range=(-1, 1), y_range=(-1, 1), resolution=256):
    """
    Plotting 2D functions in 3D.

    Parameters:
    - function (matrix): 2D function
    - *args: e.g. time step of heat kernel
    - x_range (tuple): upper and lower limits of the x-axis
    - y_range (tuple): upper and lower limits of the y-axis
    - resolution (int): resolution of the function, e.g. colour intensities of an image

    Returns:
    - None:This function displays a plot and does not return any value.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y, *args)

    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Integrand Function")

    plt.show()


def compare_3d_images(image1, image2):
    """
    Plotting images as functions in 3D.

    Parameters:
    - image1 (matrix): matrix representation of an image
    - image2 (matrix): matrix representation of an image

    Returns:
    - None:This function displays a plot and does not return any value.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
    )

    x = np.linspace(-1, 1, 256)
    y = np.linspace(-1, 1, 256)
    X, Y = np.meshgrid(x, y)

    Z1 = image1
    Z2 = image2

    ax1.plot_surface(X, Y, Z1, cmap="viridis", edgecolor="none")
    ax2.plot_surface(X, Y, Z2, cmap="viridis", edgecolor="none")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Function 1")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Function 2")

    plt.tight_layout()
    plt.show()


def plot_input(img, title):
    """
    Function to plot a grayscale image with a title.

    Parameters:
    - image (matrix): matrix representation of an image
    - title (string): title of the image

    Returns:
    - None: This function displays a plot and does not return any value.
    """
    plt.imshow(img, cmap="gray")
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def sub_plot(img1, img2, img3, title):
    """
    Function to plot the stages of the reconstruction process of the image.
    """
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img1, cmap="gray")
    plt.subplot(132)
    plt.imshow(img2, cmap="gray")
    plt.subplot(133)
    plt.imshow(img3, cmap="gray")
    plt.show()


def subplot_2(image1, image2):
    """
    Plotting two images side by side.

    Parameters:
    - image1 (matrix): matrix representation of an image.
    - image1 (matrix): matrix representation of an image.

    Returns:
    - None: This function displays a plot and does not return any value.
    """
    fig = plt.figure(figsize=(10, 10))

    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)

    plt.imshow(image1, cmap="gray")
    plt.axis("off")
    plt.title("Image 1")

    fig.add_subplot(rows, columns, 2)

    plt.imshow(image2, cmap="gray")
    plt.axis("off")
    plt.title("Image 2")


def compare_norm_histograms(image1, image2):
    """
    Plot histograms of the two given images side by side.

    Parameters:
    - image1 (matrix): matrix representation of an image.
    - image2 (matrix): matrix representation of an image.

    Returns:
    - None: This function displays a plot and does not return any value.
    """
    plt.hist(image1.ravel(), bins=256, color="blue", alpha=0.7, label="Image 1")
    plt.hist(image2.ravel(), bins=256, color="red", alpha=0.7, label="Image 2")
    plt.title("Histogram of Two Images")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def norm_histogram(image1):
    """
    Plot a histogram of the a given images.

    Parameters:
    - image (matrix): matrix representation of an image.

    Returns:
    - None: This function displays a plot and does not return any value."""
    plt.hist(image1.ravel(), bins=256, color="blue", alpha=0.7, label="Image 1")
    plt.title("Histogram of Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def plot_images(img, step):
    """
    Plot every k'th image

    Parameters:
    - img (list): input list of images
    - step (int): plot every 'step' size image

    Returns:
    - None: This function displays a plot and does not return any values
    """
    for i in range(0, (len(img)), step):
        plot_input(img[i], i)
