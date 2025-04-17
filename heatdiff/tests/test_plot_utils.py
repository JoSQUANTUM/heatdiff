"""Tests for visualization plot utilities."""

import numpy as np
import pytest

from heatdiff.visualization.plot_utils import (
    compare_3d_images,
    compare_norm_histograms,
    plot_3d,
    plot_heat_kernels,
)


@pytest.fixture
def sample_kernel():
    """Simple 3x3 kernel for testing."""
    return np.array([[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]])


@pytest.fixture
def sample_image():
    """Simple 5x5 test image."""
    return np.random.rand(5, 5)


def test_plot_heat_kernels(sample_kernel):
    """Test heat kernel plotting function."""

    def heat_kernel(t, n):
        return np.ones((n, n)) * t

    def heat_approx(t, n):
        return np.ones((n, n)) * t * 0.9

    # Should run without errors
    plot_heat_kernels(heat_kernel, heat_approx, t=0.1, n=3)


def test_plot_3d():
    """Test 3D plotting function."""

    def test_func(x, y):
        return np.sin(x) + np.cos(y)

    # Should run without errors
    plot_3d(test_func)


def test_compare_3d_images(sample_image):
    """Test 3D image comparison."""
    img1 = sample_image
    img2 = sample_image * 0.5

    # Should run without errors
    compare_3d_images(img1, img2)


def test_compare_norm_histograms(sample_image):
    """Test normalized histogram comparison."""
    img1 = sample_image
    img2 = sample_image * 0.8

    # Should run without errors
    compare_norm_histograms(img1, img2)
