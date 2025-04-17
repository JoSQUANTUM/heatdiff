"""Visualization tools for heat diffusion analysis."""

from .plot import show_image
from .plot_utils import (
    compare_3d_images,
    compare_norm_histograms,
    plot_3d,
    plot_heat_kernels,
)

__all__ = [
    "show_image",
    "plot_heat_kernels",
    "plot_3d",
    "compare_3d_images",
    "compare_norm_histograms",
]
