"""
Test basic package functionality.
"""

import pytest
from deep_learning_final_assignment import __version__, __author__, __email__


class TestPackageInfo:
    """Test package metadata and basic imports."""

    def test_version(self):
        """Test that version is properly defined."""
        assert __version__ == "0.1.0"

    def test_author(self):
        """Test that author is properly defined."""
        assert __author__ == "Istvan Peter Jaray"

    def test_email(self):
        """Test that email is properly defined."""
        assert __email__ == "istvanpeterjaray@gmail.com"

    def test_package_import(self):
        """Test that the package can be imported without errors."""
        import deep_learning_final_assignment

        assert deep_learning_final_assignment is not None


@pytest.mark.unit
def test_basic_imports():
    """Test that basic Python libraries can be imported."""
    import pandas as pd
    import requests
    import httpx

    assert pd is not None
    assert requests is not None
    assert httpx is not None


@pytest.mark.integration
def test_langchain_import():
    """Test that langchain can be imported (this might be slow)."""
    import langchain

    assert langchain is not None
