"""Tests for heat equation processing functions."""

import numpy as np
import pytest

from heatdiff.processing.heat_equations import (
    bwd_heat_equation,
    heat_equation,
    heat_semigroup,
    reverse_heat_equation,
)


@pytest.fixture
def sample_kernel():
    """Simple 3x3 kernel for testing."""
    return np.array([[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]])


@pytest.fixture
def sample_image():
    """Simple 5x5 test image."""
    return np.random.rand(5, 5)


def test_bwd_heat_equation(sample_kernel, sample_image):
    """Test backward heat equation produces expected outputs."""
    results = bwd_heat_equation(sample_kernel, sample_image, 3)
    assert len(results) == 4  # Initial + 3 steps
    assert all(isinstance(img, np.ndarray) for img in results)
    assert all(img.shape == sample_image.shape for img in results)


def test_heat_equation(sample_kernel, sample_image):
    """Test forward heat equation produces expected outputs."""
    results = heat_equation(sample_kernel, sample_image, 3)
    assert len(results) == 4
    assert all(isinstance(img, np.ndarray) for img in results)
    assert all(img.shape == sample_image.shape for img in results)


def test_heat_semigroup(sample_kernel, sample_image):
    """Test heat semigroup produces expected outputs."""
    results = heat_semigroup(sample_kernel, sample_image, 2)
    assert len(results) == 3
    assert all(isinstance(img, np.ndarray) for img in results)
    assert all(img.shape == sample_image.shape for img in results)


def test_reverse_heat_equation(sample_kernel, sample_image):
    """Test reverse heat equation produces expected outputs."""
    heat, conv = reverse_heat_equation(sample_kernel, sample_image, 2)
    assert len(heat) == 3
    assert len(conv) == 2
    assert all(isinstance(arr, np.ndarray) for arr in heat + conv)
