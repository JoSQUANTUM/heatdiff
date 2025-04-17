"""Tests for segmentation algorithms. Simple images require a high lambda"""

import numpy as np
import pytest

from heatdiff.processing.color_segmentation import ColorSegmenter
from heatdiff.processing.jacobi_segmentation import JacobiThetaSegmenter
from heatdiff.processing.kmeans_segmentation import KMeansSegmenter
from heatdiff.processing.segmentation import EsedogluTsaiSegmenter


@pytest.fixture
def test_image():
    """Create a simple test image with two regions."""
    img = np.zeros((32, 32))
    img[8:24, 8:24] = 1  # Square in center
    return img


@pytest.fixture
def test_mask():
    """Create a test mask covering part of the test image."""
    mask = np.zeros((32, 32))
    mask[12:20, 12:20] = 1  # Smaller square
    return mask


def test_segmenter_initialization():
    """Test segmenter initialization with parameters."""
    segmenter = EsedogluTsaiSegmenter(lambda_param=0.001, h=1.0)
    assert segmenter.lambda_param == 0.001
    assert segmenter.h == 1.0


def test_basic_segmentation(test_image, test_mask):
    """Test basic segmentation on synthetic image."""
    segmenter = EsedogluTsaiSegmenter(lambda_param=25)
    result = segmenter.segment(test_image, test_mask, max_iter=10)

    assert result.shape == test_mask.shape
    # Check center region is mostly segmented
    center_mean = np.mean(result[8:24, 8:24])
    assert center_mean > 0.5
    # Check border region is mostly not segmented
    border_mean = np.mean(result[:8])
    assert border_mean < 0.5


def test_convergence(test_image, test_mask):
    """Test that segmentation converges."""
    segmenter = EsedogluTsaiSegmenter(lambda_param=25)
    result1 = segmenter.segment(test_image, test_mask, max_iter=5)
    result2 = segmenter.segment(test_image, test_mask, max_iter=10)

    # Should have more segmentation with more steps
    assert np.mean(result2[8:24, 8:24]) >= np.mean(result1[8:24, 8:24])


def test_jacobi_segmenter(test_image, test_mask):
    """Test Jacobi-Theta kernel segmentation."""
    segmenter = JacobiThetaSegmenter(lambda_param=25)
    result, _ = segmenter.segment(test_image, test_mask, max_iter=10)

    assert result.shape == test_mask.shape
    assert np.mean(result[8:24, 8:24]) > 0.5


def test_kmeans_segmenter(test_image, test_mask):
    """Test K-means enhanced segmentation."""
    segmenter = KMeansSegmenter(lambda_param=25)
    result, _ = segmenter.segment(test_image, test_mask, max_iter=10)

    assert result.shape == test_mask.shape
    assert np.mean(result[8:24, 8:24]) > 0.5


@pytest.fixture
def color_test_image():
    """Create a simple color test image."""
    img = np.zeros((32, 32, 3))
    img[8:24, 8:24] = [1, 0.5, 0.5]  # Pink square
    return img


def test_color_segmenter(color_test_image, test_mask):
    """Test color image segmentation."""
    segmenter = ColorSegmenter(lambda_param=25)
    result, _ = segmenter.segment(color_test_image, test_mask, max_iter=10)

    assert result.shape == test_mask.shape
    assert np.mean(result[8:24, 8:24]) > 0.5
