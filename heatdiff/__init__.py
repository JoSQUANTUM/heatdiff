"""HeatDiff - Image processing using heat equation methods."""

from .image.compression import hermite_compression, jacobi_compression
from .image.image import add_noise, load_image
from .processing.color_segmentation import ColorSegmenter
from .processing.diffusion import (
    reverse_diffusion,
    reverse_ou,
    reverse_wiener_1d,
    reverse_wiener_img,
    wiener_1d,
)
from .processing.heat_equations import (
    bwd_heat_equation,
    heat_equation,
    heat_semigroup,
    reverse_heat_equation,
)
from .processing.jacobi_segmentation import JacobiThetaSegmenter
from .processing.kmeans_segmentation import KMeansSegmenter
from .processing.segmentation import EsedogluTsaiSegmenter
from .visualization import (
    compare_3d_images,
    compare_norm_histograms,
    plot_3d,
    plot_heat_kernels,
    show_image,
)

__version__ = "0.1.0"
__all__ = [
    "EsedogluTsaiSegmenter",
    "JacobiThetaSegmenter",
    "KMeansSegmenter",
    "ColorSegmenter",
    "bwd_heat_equation",
    "heat_equation",
    "reverse_heat_equation",
    "heat_semigroup",
    "jacobi_compression",
    "hermite_compression",
    "load_image",
    "add_noise",
    "show_image",
    "plot_heat_kernels",
    "plot_3d",
    "compare_3d_images",
    "compare_norm_histograms",
    "reverse_diffusion",
    "reverse_ou",
    "reverse_wiener_1d",
    "reverse_wiener_img",
    "wiener_1d",
]
