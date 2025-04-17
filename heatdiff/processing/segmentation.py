"""Esedoglu-Tsai image segmentation algorithm. Python version of https://github.com/nathandking/ImageSegmentation/blob/master/EsedogluTsai/CameraMan.m"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


class EsedogluTsaiSegmenter:
    """Implements the Esedoglu-Tsai segmentation algorithm using heat diffusion.

    Attributes:
        lambda_param (float): Tunable parameter for segmentation strength
        h (float): Space step size
        dt (float): Time step size derived from h
        k (float): Time step size for diffusion part (same as dt)
    """

    def __init__(self, lambda_param: float = 0.00025, h: float = 1.0):
        """Initialize segmenter with parameters.

        Args:
            lambda_param: Tunable parameter controlling segmentation strength
            h: Space step size for discretization
        """
        self.lambda_param = lambda_param
        self.h = h
        self.dt = 0.8 * h**2  # time step size
        self.k = self.dt  # time step size for diffusion part

    def create_matrices(self, size: int) -> NDArray:
        """Create sparse matrices for finite difference discretization.

        Args:
            size: Dimension of the input image (assumes square)

        Returns:
            2D finite difference matrix in CSC sparse format
        """
        E = diags([1], [1], shape=(size, size)).tocsc()
        Imat = eye(size).tocsc()
        A1D = E + E.T - 2 * Imat
        # Adjust boundary conditions
        A1D[size - 1, size - 2] = 2
        A1D[0, 1] = 2
        return kron(A1D, Imat) + kron(Imat, A1D)

    def segment(
        self,
        image: NDArray,
        initial_guess: Optional[NDArray] = None,
        max_iter: int = 20,
    ) -> NDArray:
        """Perform segmentation on input image.

        Args:
            image: Input grayscale image as 2D numpy array
            initial_guess: Optional initial segmentation mask (same size as image)
            max_iter: Maximum number of iterations to run

        Returns:
            Segmented binary image as 2D numpy array
        """
        size = image.shape[0]
        A2D = self.create_matrices(size)
        C = (self.lambda_param / np.sqrt(np.pi * self.k)) * np.mean(image)
        Amat = (1 + self.k * C) * eye(size**2) - A2D

        # Initialize with circle if no guess provided
        if initial_guess is None:
            initial_guess = self._create_circle_guess(size)

        u = initial_guess.flatten()

        for _ in range(max_iter):
            u = self._diffusion_step(u, image, Amat, C, size)
            u = self._threshold(u, size)

        return u.reshape(size, size)

    def _create_circle_guess(self, size: int, radius: float = 10) -> NDArray:
        """Create initial circle segmentation guess.

        Args:
            size: Dimension of the image (assumes square)
            radius: Radius of the initial circle

        Returns:
            Initial binary mask with circle
        """
        guess = np.zeros((size, size))
        center = size / 2
        x = y = np.arange(1, size + 1)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        guess[rho <= radius] = 1
        return guess

    def _diffusion_step(
        self, u: NDArray, image: NDArray, Amat: NDArray, C: float, size: int
    ) -> NDArray:
        """Perform single diffusion step with Neumann boundary conditions."""
        time = 0
        while time < self.dt:
            # Apply Neumann boundary conditions
            u[0] = u[1]  # du/dx = 0 at x = 0
            u[size - 1] = u[size - 2]  # du/dx = 0 at x = N-1
            u[::size] = u[1::size]  # du/dy = 0 at y = 0
            u[size - 1 :: size] = u[size - 2 :: size]  # du/dy = 0 at y = N-1

            fvec = image.flatten()
            C1 = np.mean(fvec[u >= 0.5])
            C2 = np.mean(fvec[u < 0.5])

            A = (self.lambda_param / np.sqrt(np.pi * self.k)) * (
                (C1 - fvec) ** 2 + (C2 - fvec) ** 2
            )
            B = (self.lambda_param / np.sqrt(np.pi * self.k)) * (C2 - fvec) ** 2

            u = spsolve(Amat, u + self.k * (C - A) * u + B)
            time += self.k
        return u

    def _threshold(self, u: NDArray, size: int) -> NDArray:
        """Apply binary threshold to segmentation result."""
        u[u <= 0.5] = 0
        u[u > 0.5] = 1
        return u
