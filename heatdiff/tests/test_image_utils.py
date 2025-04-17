"""Tests for image utilities."""

import os

import numpy as np
import pytest

from heatdiff.utils.image import add_noise, load_image


@pytest.fixture
def test_grayscale_image():
    """Create a simple grayscale test image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 128  # Gray square in center
    return img


@pytest.fixture
def test_color_image():
    """Create a simple color test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [255, 128, 0]  # Orange square in center
    return img


@pytest.mark.integration
def test_load_image_grayscale():
    """Test loading grayscale image from file."""
    test_img = os.path.join(os.path.dirname(__file__), "test_data/images/cameraman.jpg")
    if not os.path.exists(test_img):
        pytest.skip("Test image not found")
    img = load_image(test_img, grayscale=True)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert 0 <= img.min() <= img.max() <= 255


@pytest.mark.integration
def test_load_image_color():
    """Test loading color image from file."""
    test_img = os.path.join(os.path.dirname(__file__), "test_data/images/flowers.jpg")
    if not os.path.exists(test_img):
        pytest.skip("Test image not found")
    img = load_image(test_img, grayscale=False)
    assert img.ndim == 3
    assert img.shape[2] == 3  # RGB channels
    assert img.dtype == np.uint8


def test_add_noise(test_grayscale_image):
    """Test noise addition to image."""
    img = np.zeros((100, 100), dtype=np.uint8) + 128
    noisy = add_noise(img, mean=0, std=25)

    assert noisy.shape == img.shape
    assert noisy.dtype == np.uint8
    assert np.any(noisy != img)  # Should actually add noise
    assert 0 <= noisy.min() <= noisy.max() <= 255


def test_noise_preserves_range():
    """Test noise doesn't cause values to wrap around."""
    black = np.zeros((10, 10), dtype=np.uint8)
    white = np.zeros((10, 10), dtype=np.uint8) + 255

    noisy_black = add_noise(black, std=25)
    noisy_white = add_noise(white, std=25)

    assert noisy_black.min() >= 0
    assert noisy_white.max() <= 255
