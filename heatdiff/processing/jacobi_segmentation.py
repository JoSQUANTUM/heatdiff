"""Jacobi-Theta kernel based segmentation implementation."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal


class JacobiThetaSegmenter:
    """Implements Esedoglu-Tsai segmentation using Jacobi-Theta kernel."""

    def __init__(self, lambda_param: float = 0.00025, dt: float = 0.5):
        """
        Initialize segmenter with parameters.

        Args:
            lambda_param: Tunable parameter for segmentation
            dt: Time step size
        """
        self.lambda_param = lambda_param
        self.dt = dt
        self.k = dt / 3  # Split into 3 steps
        self.conv_thresh = 1e-4  # Convergence threshold

    def create_jacobi_theta_kernel(self, size: int) -> Tuple[NDArray, float]:
        """
        Create 1D Jacobi-Theta kernel for heat diffusion.

        Args:
            size: Kernel size (should be odd)

        Returns:
            Tuple of (1D kernel, optimal_time)
        """
        x_range = np.arange(-size // 2, size // 2 + 1) / size
        num_terms = 2 * size  # Number of terms in summation

        optimal_time = np.sqrt(
            np.abs(np.log(self.conv_thresh)) / (np.pi**2 * num_terms)
        )

        n_range = np.arange(1, num_terms + 1).reshape(-1, 1)
        kernel = 1 + 2 * np.sum(
            np.exp(-self.k * np.pi * n_range**2)
            * np.cos(2 * n_range * np.pi * x_range),
            axis=0,
        )
        kernel /= np.sum(kernel)  # Normalize

        return kernel, optimal_time

    def diffuse(self, u: NDArray, kernel: NDArray) -> NDArray:
        """
        Apply Jacobi-Theta kernel diffusion with Neumann BCs.

        Args:
            u: Input segmentation mask
            kernel: 1D Jacobi-Theta kernel

        Returns:
            Diffused mask
        """
        # Apply Neumann boundary conditions
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

        # Apply separable convolution
        temp = signal.convolve(u, kernel.reshape(1, -1), mode="same")
        return signal.convolve(temp, kernel.reshape(-1, 1), mode="same")

    def react(self, u_diff: NDArray, image: NDArray) -> NDArray:
        """
        Perform reaction step.

        Args:
            u_diff: Diffused mask
            image: Input image

        Returns:
            Reacted mask
        """
        fvec = image.flatten()
        z = u_diff.flatten()

        # Apply Neumann BCs
        z_matrix = z.reshape(u_diff.shape)
        z_matrix[0, :] = z_matrix[1, :]
        z_matrix[-1, :] = z_matrix[-2, :]
        z_matrix[:, 0] = z_matrix[:, 1]
        z_matrix[:, -1] = z_matrix[:, -2]
        z = z_matrix.flatten()

        scale = self.lambda_param / np.sqrt(np.pi * self.k)
        C1 = np.mean(fvec[z >= 0.5])
        C2 = np.mean(fvec[z < 0.5])

        A = scale * ((C1 - fvec) ** 2 + (C2 - fvec) ** 2)
        B = scale * (C2 - fvec) ** 2
        C_val_pre = scale * np.mean(fvec)

        return z + self.k * (B + (C_val_pre - A) * z)

    def segment(
        self, image: NDArray, initial_guess: NDArray, max_iter: int = 20
    ) -> Tuple[NDArray, list]:
        """
        Perform segmentation.

        Args:
            image: Input image
            initial_guess: Initial segmentation mask
            max_iter: Maximum iterations

        Returns:
            Tuple of (final segmentation, energy history)
        """
        kernel, _ = self.create_jacobi_theta_kernel(max(3, image.shape[0] // 5))
        u = initial_guess.copy()
        previous_u = np.zeros_like(u)
        energies = []

        for _iteration in range(max_iter):
            # Diffusion step
            u_diff = self.diffuse(u, kernel)

            # Reaction step
            z = self.react(u_diff, image)
            z = z.reshape(u.shape)  # Ensure same shape as input

            # Assignment step
            u = z.copy()
            u[u <= 0.5] = 0
            u[u > 0.5] = 1

            # Convergence check
            difference = np.linalg.norm(u - previous_u)
            relative_diff = (
                difference / np.linalg.norm(u) if np.linalg.norm(u) > 0 else 0
            )
            energies.append(relative_diff)

            if relative_diff < self.conv_thresh:
                break

            previous_u = u.copy()

        return u, energies
