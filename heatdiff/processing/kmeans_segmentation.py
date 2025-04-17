"""K-means based segmentation implementation."""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .jacobi_segmentation import JacobiThetaSegmenter


class KMeansSegmenter(JacobiThetaSegmenter):
    """Implements Esedoglu-Tsai segmentation using K-means in reaction step."""

    def react(self, u_diff: NDArray, image: NDArray) -> NDArray:
        """
        Perform reaction step using K-means clustering.

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

        # Normalize pixel intensities for K-means
        scaler = MinMaxScaler()
        pixels_normalized = scaler.fit_transform(fvec.reshape(-1, 1))

        # Apply K-means with initial centers based on current segmentation
        kmeans = KMeans(
            n_clusters=2,
            init=np.array(
                [
                    [np.mean(pixels_normalized[z >= 0.5])],
                    [np.mean(pixels_normalized[z < 0.5])],
                ]
            ),
            n_init=1,
        )
        kmeans.fit(pixels_normalized)
        C1, C2 = scaler.inverse_transform(kmeans.cluster_centers_).flatten()

        scale = self.lambda_param / np.sqrt(np.pi * self.k)
        A = scale * ((C1 - fvec) ** 2 + (C2 - fvec) ** 2)
        B = scale * (C2 - fvec) ** 2
        C_val_pre = scale * np.mean(fvec)

        return z + self.k * (B + (C_val_pre - A) * z)
