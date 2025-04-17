import heatdiff


def test_package_import():
    """Test that the package can be imported."""
    assert heatdiff is not None


def test_version_exists():
    """Test that __version__ exists in the package."""
    assert hasattr(heatdiff, "__version__")
