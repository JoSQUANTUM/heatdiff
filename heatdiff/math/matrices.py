"""Matrix creation utilities for finite difference methods."""

from numpy.typing import NDArray
from scipy.sparse import diags, eye, kron


def create_2d_finite_diff_matrix(size: int) -> NDArray:
    """Create 2D finite difference matrix for heat diffusion.

    Args:
        size: Dimension of the problem (assumes square grid)

    Returns:
        Sparse matrix in CSC format representing 2D finite differences
    """
    E = diags([1], [1], shape=(size, size)).tocsc()
    Imat = eye(size).tocsc()
    A1D = E + E.T - 2 * Imat
    # Adjust boundary conditions for Neumann
    A1D[size - 1, size - 2] = 2
    A1D[0, 1] = 2
    return kron(A1D, Imat) + kron(Imat, A1D)
