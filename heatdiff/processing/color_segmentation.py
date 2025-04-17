"""Color image segmentation implementation."""

import numpy as np
from numpy.typing import NDArray

from .jacobi_segmentation import JacobiThetaSegmenter


class ColorSegmenter(JacobiThetaSegmenter):
    """Implements Esedoglu-Tsai segmentation for color images."""

    def react(self, u_diff: NDArray, image: NDArray) -> NDArray:
        """
        Perform reaction step for color images.

        Args:
            u_diff: Diffused mask (2D)
            image: Input color image (3D)

        Returns:
            Reacted mask
        """
        # Flatten spatial dimensions
        z = u_diff.flatten()
        fvec = image.reshape(-1, image.shape[-1])  # (N*M, channels)

        # Apply Neumann BCs
        z_matrix = z.reshape(u_diff.shape)
        z_matrix[0, :] = z_matrix[1, :]
        z_matrix[-1, :] = z_matrix[-2, :]
        z_matrix[:, 0] = z_matrix[:, 1]
        z_matrix[:, -1] = z_matrix[:, -2]
        z = z_matrix.flatten()

        # Compute means for each channel
        C1 = np.mean(fvec[z >= 0.5], axis=0)
        C2 = np.mean(fvec[z < 0.5], axis=0)

        scale = self.lambda_param / np.sqrt(np.pi * self.k)
        A = scale * (np.sum((C1 - fvec) ** 2 + (C2 - fvec) ** 2, axis=1))
        B = scale * np.sum((C2 - fvec) ** 2, axis=1)
        C_val_pre = scale * np.mean(fvec)

        return z + self.k * (B + (C_val_pre - A) * z)
