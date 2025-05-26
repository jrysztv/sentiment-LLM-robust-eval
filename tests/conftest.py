"""
Pytest configuration and fixtures for the test suite.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from deep_learning_final_assignment.core.models.base import ModelResponse
from deep_learning_final_assignment.core.data.loaders import DataSample, SentimentLabels
from deep_learning_final_assignment.core.prompts import (
    SentimentPrompts,
    PromptPerturbator,
)
from deep_learning_final_assignment.core.config import load_experiment_config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_data_dir():
    """Provide path to sample data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    test_vars = {
        "TEST_MODE": "true",
        "OPENAI_API_KEY": "test_openai_key",
        "DEBUG": "true",
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


@pytest.fixture
def sample_data_samples():
    """Create sample DataSample objects for testing."""
    return [
        DataSample(
            text="This movie is absolutely terrible!",
            label="Very Negative",
            encoding=-3,
            metadata={"text_length": 32, "word_count": 5},
        ),
        DataSample(
            text="I didn't like this film.",
            label="Negative",
            encoding=-2,
            metadata={"text_length": 24, "word_count": 5},
        ),
        DataSample(
            text="The movie was okay, nothing special.",
            label="Neutral",
            encoding=0,
            metadata={"text_length": 36, "word_count": 6},
        ),
        DataSample(
            text="I enjoyed watching this movie!",
            label="Positive",
            encoding=2,
            metadata={"text_length": 31, "word_count": 5},
        ),
        DataSample(
            text="This is the best movie ever made!",
            label="Very Positive",
            encoding=3,
            metadata={"text_length": 34, "word_count": 7},
        ),
    ]


@pytest.fixture
def sample_model_responses():
    """Create sample ModelResponse objects for testing."""
    return [
        ModelResponse(
            content='{"sentiment": "Very Negative"}',
            model_name="test_model",
            prompt_used="Test prompt 1",
            metadata={"temperature": 0.1},
        ),
        ModelResponse(
            content='{"sentiment": "Negative"}',
            model_name="test_model",
            prompt_used="Test prompt 2",
            metadata={"temperature": 0.1},
        ),
        ModelResponse(
            content='{"sentiment": "Neutral"}',
            model_name="test_model",
            prompt_used="Test prompt 3",
            metadata={"temperature": 0.1},
        ),
        ModelResponse(
            content='{"sentiment": "Positive"}',
            model_name="test_model",
            prompt_used="Test prompt 4",
            metadata={"temperature": 0.1},
        ),
        ModelResponse(
            content='{"sentiment": "Very Positive"}',
            model_name="test_model",
            prompt_used="Test prompt 5",
            metadata={"temperature": 0.1},
        ),
    ]


@pytest.fixture
def sentiment_prompts():
    """Create SentimentPrompts instance for testing."""
    return SentimentPrompts("test_sentiment_classification")


@pytest.fixture
def prompt_perturbator(sentiment_prompts):
    """Create PromptPerturbator instance for testing."""
    return PromptPerturbator(sentiment_prompts)


@pytest.fixture
def test_config():
    """Load test configuration."""
    try:
        config_path = Path(__file__).parent.parent / "config" / "test_config.json"
        return load_experiment_config(str(config_path))
    except Exception:
        # Fallback to default config if test config not found
        return load_experiment_config()


@pytest.fixture
def mock_openai_model(test_config):
    """Create a mock OpenAI model for testing."""
    openai_config = test_config.get_openai_config()
    mock_model = Mock()
    mock_model.model_name = openai_config.model_name
    mock_model.is_available = True

    def mock_generate(prompt, **kwargs):
        return ModelResponse(
            content='{"sentiment": "Positive"}',
            model_name=openai_config.model_name,
            prompt_used=prompt,
            metadata={"temperature": kwargs.get("temperature", 0.1)},
        )

    def mock_batch_generate(prompts, **kwargs):
        return [mock_generate(prompt, **kwargs) for prompt in prompts]

    mock_model.generate = mock_generate
    mock_model.batch_generate = mock_batch_generate

    return mock_model


@pytest.fixture
def mock_ollama_model(test_config):
    """Create a mock Ollama model for testing."""
    ollama_config = test_config.get_ollama_config()
    mock_model = Mock()
    mock_model.model_name = ollama_config.model_name
    mock_model.is_available = True

    def mock_generate(prompt, **kwargs):
        return ModelResponse(
            content='{"sentiment": "Neutral"}',
            model_name=ollama_config.model_name,
            prompt_used=prompt,
            metadata={"temperature": kwargs.get("temperature", 0.1)},
        )

    def mock_batch_generate(prompts, **kwargs):
        return [mock_generate(prompt, **kwargs) for prompt in prompts]

    mock_model.generate = mock_generate
    mock_model.batch_generate = mock_batch_generate

    return mock_model


@pytest.fixture
def mock_sst5_data():
    """Create mock SST-5 dataset for testing."""
    import pandas as pd

    data = [
        {"text": "This movie is terrible!", "label": 0},  # Very Negative
        {"text": "I don't like it.", "label": 1},  # Negative
        {"text": "It's okay.", "label": 2},  # Neutral
        {"text": "I like this movie.", "label": 3},  # Positive
        {"text": "Amazing film!", "label": 4},  # Very Positive
    ]

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def session_setup():
    """Session-wide setup for expensive operations."""
    print("\nSetting up test session...")
    yield
    print("\nTearing down test session...")


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API access"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama server"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests that don't have integration/slow markers
        if not any(
            mark.name in ["integration", "slow", "requires_api", "requires_ollama"]
            for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
