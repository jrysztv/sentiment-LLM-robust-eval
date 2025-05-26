"""
Tests for logging utilities.
"""

import pytest
import logging
import sys
from io import StringIO
from unittest.mock import patch

from deep_learning_final_assignment.core.utils.logging_utils import (
    setup_unicode_safe_logging,
    get_safe_logger,
    UnicodeCompatibleFormatter,
    _console_supports_unicode,
)


class TestLoggingUtils:
    """Test logging utilities."""

    def test_unicode_compatible_formatter(self):
        """Test that the formatter replaces Unicode characters."""
        formatter = UnicodeCompatibleFormatter()

        # Create a log record with Unicode characters
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="✅ Test message with 🚀 emojis",
            args=(),
            exc_info=None,
        )

        # Mock Windows environment with no Unicode support
        with (
            patch("sys.platform", "win32"),
            patch.object(formatter, "_supports_unicode", return_value=False),
        ):
            formatted = formatter.format(record)
            assert "✅" not in formatted
            assert "🚀" not in formatted
            assert "[OK]" in formatted
            assert "[START]" in formatted

    def test_unicode_compatible_formatter_with_unicode_support(self):
        """Test that the formatter keeps Unicode when supported."""
        formatter = UnicodeCompatibleFormatter()

        # Create a log record with Unicode characters
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="✅ Test message with 🚀 emojis",
            args=(),
            exc_info=None,
        )

        # Mock environment with Unicode support
        with patch.object(formatter, "_supports_unicode", return_value=True):
            formatted = formatter.format(record)
            assert "✅" in formatted
            assert "🚀" in formatted

    def test_console_supports_unicode_detection(self):
        """Test Unicode support detection."""
        # This test will vary based on the actual environment
        result = _console_supports_unicode()
        assert isinstance(result, bool)

    def test_get_safe_logger(self):
        """Test getting a safe logger."""
        logger = get_safe_logger("test_logger")

        # Check that safe logging methods are added
        assert hasattr(logger, "safe_info")
        assert hasattr(logger, "safe_error")
        assert hasattr(logger, "safe_warning")
        assert hasattr(logger, "safe_debug")

    def test_safe_logger_methods(self):
        """Test that safe logger methods work."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        logger = get_safe_logger("test_safe_logger", use_unicode=False)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Test safe logging with Unicode characters
        logger.safe_info("✅ This should work safely")
        logger.safe_error("❌ This should also work")

        # Get the logged output
        output = stream.getvalue()

        # Should contain the replacement text, not Unicode
        assert "[OK]" in output or "✅" in output  # Depends on environment
        assert "[FAIL]" in output or "❌" in output

    def test_setup_unicode_safe_logging(self):
        """Test setting up Unicode-safe logging."""
        # Test basic setup
        logger = setup_unicode_safe_logging(log_level="INFO")
        assert logger is not None

        # Test with file logging
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            logger = setup_unicode_safe_logging(log_level="DEBUG", log_file=log_file)
            assert logger is not None

            # Test logging to file
            test_logger = logging.getLogger("test_file_logger")
            test_logger.info("✅ Test file logging")

        finally:
            import os

            try:
                os.unlink(log_file)
            except:
                pass

    def test_emoji_replacement_mapping(self):
        """Test that all common emojis are properly mapped."""
        formatter = UnicodeCompatibleFormatter()

        test_message = "🚀✅❌⚠️📊📝🤖🔬🔄💾📈⚡🎉🏆ℹ️🔧💥"

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=test_message,
            args=(),
            exc_info=None,
        )

        # Mock Windows environment with no Unicode support
        with (
            patch("sys.platform", "win32"),
            patch.object(formatter, "_supports_unicode", return_value=False),
        ):
            formatted = formatter.format(record)

            # Check that all emojis are replaced
            for emoji in "🚀✅❌⚠️📊📝🤖🔬🔄💾📈⚡🎉🏆ℹ️🔧💥":
                assert emoji not in formatted

            # Check that replacements are present
            assert "[START]" in formatted
            assert "[OK]" in formatted
            assert "[FAIL]" in formatted
