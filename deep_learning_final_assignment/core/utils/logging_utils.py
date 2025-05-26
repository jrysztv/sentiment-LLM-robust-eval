"""
Logging utilities with Unicode support for Windows.
"""

import logging
import sys
import os
from typing import Optional


class UnicodeCompatibleFormatter(logging.Formatter):
    """Formatter that handles Unicode characters safely on Windows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map of Unicode emojis to ASCII alternatives
        self.emoji_map = {
            "🚀": "[START]",
            "✅": "[OK]",
            "❌": "[FAIL]",
            "⚠️": "[WARN]",
            "📊": "[DATA]",
            "📝": "[NOTE]",
            "🤖": "[AI]",
            "🔬": "[TEST]",
            "🔄": "[PROC]",
            "💾": "[SAVE]",
            "📈": "[CALC]",
            "⚡": "[FAST]",
            "🎉": "[DONE]",
            "🏆": "[BEST]",
            "ℹ️": "[INFO]",
            "🔧": "[CONF]",
            "💥": "[ERR]",
        }

    def format(self, record):
        """Format the record, replacing Unicode characters if needed."""
        formatted = super().format(record)

        # Check if we're on Windows and the console doesn't support Unicode
        if sys.platform == "win32" and not self._supports_unicode():
            for emoji, replacement in self.emoji_map.items():
                formatted = formatted.replace(emoji, replacement)

        return formatted

    def _supports_unicode(self) -> bool:
        """Check if the current console supports Unicode output."""
        try:
            # Try to encode a Unicode character
            "✅".encode(sys.stdout.encoding or "utf-8")
            return True
        except (UnicodeEncodeError, LookupError):
            return False


def setup_unicode_safe_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_unicode: Optional[bool] = None,
) -> logging.Logger:
    """
    Setup logging with Unicode safety for Windows.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_unicode: Force Unicode on/off (None = auto-detect)

    Returns:
        Configured logger
    """
    # Determine if we should use Unicode
    if use_unicode is None:
        use_unicode = not (sys.platform == "win32" and not _console_supports_unicode())

    # Create formatter
    if use_unicode:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = UnicodeCompatibleFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        # File handler can always use Unicode (UTF-8)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    return logging.getLogger(__name__)


def _console_supports_unicode() -> bool:
    """Check if the Windows console supports Unicode output."""
    try:
        # Try to encode a Unicode character to the console encoding
        test_char = "✅"
        encoding = sys.stdout.encoding or "cp1252"
        test_char.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def get_safe_logger(name: str, use_unicode: Optional[bool] = None) -> logging.Logger:
    """
    Get a logger that's safe for Unicode output.

    Args:
        name: Logger name
        use_unicode: Force Unicode on/off (None = auto-detect)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Add a custom method for safe logging
    def safe_log(level, message, *args, **kwargs):
        """Log a message with Unicode safety."""
        if use_unicode is False or (
            use_unicode is None
            and sys.platform == "win32"
            and not _console_supports_unicode()
        ):
            # Replace Unicode characters
            emoji_map = {
                "🚀": "[START]",
                "✅": "[OK]",
                "❌": "[FAIL]",
                "⚠️": "[WARN]",
                "📊": "[DATA]",
                "📝": "[NOTE]",
                "🤖": "[AI]",
                "🔬": "[TEST]",
                "🔄": "[PROC]",
                "💾": "[SAVE]",
                "📈": "[CALC]",
                "⚡": "[FAST]",
                "🎉": "[DONE]",
                "🏆": "[BEST]",
                "ℹ️": "[INFO]",
                "🔧": "[CONF]",
                "💥": "[ERR]",
            }
            for emoji, replacement in emoji_map.items():
                message = message.replace(emoji, replacement)

        logger.log(level, message, *args, **kwargs)

    # Add convenience methods
    logger.safe_info = lambda msg, *args, **kwargs: safe_log(
        logging.INFO, msg, *args, **kwargs
    )
    logger.safe_error = lambda msg, *args, **kwargs: safe_log(
        logging.ERROR, msg, *args, **kwargs
    )
    logger.safe_warning = lambda msg, *args, **kwargs: safe_log(
        logging.WARNING, msg, *args, **kwargs
    )
    logger.safe_debug = lambda msg, *args, **kwargs: safe_log(
        logging.DEBUG, msg, *args, **kwargs
    )

    return logger
