"""Tests for image compression utilities."""

import numpy as np
import pytest

from heatdiff.utils.compression import compress_image, decompress_image


@pytest.fixture
def sample_image():
    """Simple 5x5 test image."""
    return np.random.rand(5, 5)


def test_compress_image(sample_image):
    """Test compression produces valid outputs."""
    compressed, ratio = compress_image(sample_image, quality=50)

    assert isinstance(compressed, np.ndarray)
    assert compressed.shape == sample_image.shape
    assert isinstance(ratio, float)
    assert 0 <= ratio <= 1


def test_decompress_image(sample_image):
    """Test decompression produces expected shape."""
    compressed, _ = compress_image(sample_image, quality=50)
    decompressed = decompress_image(compressed, sample_image.shape)

    assert isinstance(decompressed, np.ndarray)
    assert decompressed.shape == sample_image.shape


def test_compress_invalid_quality(sample_image):
    """Test invalid quality values raise errors."""
    with pytest.raises(ValueError):
        compress_image(sample_image, quality=0)

    with pytest.raises(ValueError):
        compress_image(sample_image, quality=101)
