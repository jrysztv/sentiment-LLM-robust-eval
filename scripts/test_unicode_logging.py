#!/usr/bin/env python3
"""
Test script to demonstrate Unicode-safe logging.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_learning_final_assignment.core.utils import (
    setup_unicode_safe_logging,
    get_safe_logger,
)


def test_unicode_logging():
    """Test Unicode-safe logging functionality."""
    print("Testing Unicode-safe logging...")
    print(f"Platform: {sys.platform}")
    print(f"Console encoding: {sys.stdout.encoding}")

    # Setup Unicode-safe logging
    setup_unicode_safe_logging(log_level="INFO")

    # Get a safe logger
    logger = get_safe_logger(__name__)

    print("\n" + "=" * 50)
    print("TESTING REGULAR LOGGING (may cause Unicode errors)")
    print("=" * 50)

    try:
        import logging

        regular_logger = logging.getLogger("regular")
        regular_logger.info("🚀 Starting test with regular logger")
        regular_logger.info("✅ This might work or cause Unicode errors")
        regular_logger.error("❌ This could fail on Windows console")
        print("Regular logging completed without errors")
    except Exception as e:
        print(f"Regular logging failed: {e}")

    print("\n" + "=" * 50)
    print("TESTING SAFE LOGGING (should always work)")
    print("=" * 50)

    try:
        logger.safe_info("🚀 Starting test with safe logger")
        logger.safe_info("✅ This should work on all platforms")
        logger.safe_error("❌ This should also work safely")
        logger.safe_info("📊 Processing data...")
        logger.safe_info("🤖 AI model ready")
        logger.safe_info("⚡ Fast execution completed")
        logger.safe_info("🎉 All tests passed!")
        print("Safe logging completed successfully!")
    except Exception as e:
        print(f"Safe logging failed: {e}")

    print("\n" + "=" * 50)
    print("UNICODE SUPPORT DETECTION")
    print("=" * 50)

    from deep_learning_final_assignment.core.utils.logging_utils import (
        _console_supports_unicode,
    )

    unicode_support = _console_supports_unicode()
    print(f"Console supports Unicode: {unicode_support}")

    if unicode_support:
        print("✅ Your console supports Unicode characters!")
    else:
        print("[OK] Your console doesn't support Unicode - using ASCII replacements")


if __name__ == "__main__":
    test_unicode_logging()
