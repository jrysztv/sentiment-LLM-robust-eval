#!/usr/bin/env python3
"""
Test runner script for the Deep Learning Final Assignment project.

This script provides convenient commands to run different test suites:
- Unit tests (fast, isolated)
- Integration tests (slower, test interactions)
- All tests
- Tests with coverage
- Specific test modules

Usage:
    python scripts/run_tests.py unit           # Run only unit tests
    python scripts/run_tests.py integration    # Run only integration tests
    python scripts/run_tests.py all            # Run all tests
    python scripts/run_tests.py coverage       # Run all tests with coverage
    python scripts/run_tests.py data           # Run only data-related tests
    python scripts/run_tests.py models         # Run only model-related tests
    python scripts/run_tests.py prompts        # Run only prompt-related tests
    python scripts/run_tests.py evaluation     # Run only evaluation-related tests
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for the Deep Learning Final Assignment project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "test_type",
        choices=[
            "unit",
            "integration",
            "all",
            "coverage",
            "data",
            "models",
            "prompts",
            "evaluation",
            "fast",
            "slow",
            "api",
            "ollama",
        ],
        help="Type of tests to run",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )

    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    parser.add_argument(
        "--failfast", "-x", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Base command
    base_cmd = ["poetry", "run", "pytest"]

    # Add common options
    if args.verbose:
        base_cmd.append("-v")

    if args.parallel:
        base_cmd.extend(["-n", "auto"])

    if args.failfast:
        base_cmd.append("-x")

    # Determine test command based on type
    success = True

    if args.test_type == "unit":
        cmd = base_cmd + ["-m", "unit"]
        success = run_command(cmd, "Unit Tests")

    elif args.test_type == "integration":
        cmd = base_cmd + ["-m", "integration"]
        success = run_command(cmd, "Integration Tests")

    elif args.test_type == "all":
        cmd = base_cmd + ["tests/"]
        success = run_command(cmd, "All Tests")

    elif args.test_type == "coverage":
        cmd = base_cmd + [
            "--cov=deep_learning_final_assignment",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "tests/",
        ]
        success = run_command(cmd, "All Tests with Coverage")

        if success:
            print(f"\n📊 Coverage report generated in htmlcov/index.html")

    elif args.test_type == "data":
        cmd = base_cmd + ["tests/test_data.py"]
        success = run_command(cmd, "Data Loading Tests")

    elif args.test_type == "models":
        cmd = base_cmd + ["tests/test_models.py"]
        success = run_command(cmd, "Model Interface Tests")

    elif args.test_type == "prompts":
        cmd = base_cmd + ["tests/test_prompts.py"]
        success = run_command(cmd, "Prompt System Tests")

    elif args.test_type == "evaluation":
        cmd = base_cmd + ["tests/test_evaluation.py"]
        success = run_command(cmd, "Evaluation System Tests")

    elif args.test_type == "fast":
        cmd = base_cmd + ["-m", "not slow and not requires_api and not requires_ollama"]
        success = run_command(cmd, "Fast Tests (No External Dependencies)")

    elif args.test_type == "slow":
        cmd = base_cmd + ["-m", "slow"]
        success = run_command(cmd, "Slow Tests")

    elif args.test_type == "api":
        cmd = base_cmd + ["-m", "requires_api"]
        success = run_command(cmd, "API-Dependent Tests")

    elif args.test_type == "ollama":
        cmd = base_cmd + ["-m", "requires_ollama"]
        success = run_command(cmd, "Ollama-Dependent Tests")

    # Summary
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("- Review any test output above")
        if args.test_type == "coverage":
            print("- Open htmlcov/index.html to view detailed coverage report")
        print("- Run 'python scripts/run_baseline.py' to execute the main experiment")
    else:
        print("💥 Some tests failed!")
        print("\nTroubleshooting:")
        print("- Check the error messages above")
        print("- Ensure all dependencies are installed: poetry install")
        print("- For API tests, ensure OPENAI_API_KEY is set")
        print("- For Ollama tests, ensure Ollama server is running")
    print(f"{'=' * 60}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
