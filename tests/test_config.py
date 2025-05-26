"""
Tests for configuration management system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from deep_learning_final_assignment.core.config import (
    ModelConfig,
    OpenAIConfig,
    OllamaConfig,
    ExperimentConfig,
    EvaluationConfig,
    OutputConfig,
    ExperimentConfigLoader,
    load_experiment_config,
)


class TestModelConfig:
    """Test the ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            model_name="test_model",
            temperature=0.2,
            max_tokens=100,
        )

        assert config.model_name == "test_model"
        assert config.temperature == 0.2
        assert config.max_tokens == 100

    def test_model_config_to_dict(self):
        """Test converting ModelConfig to dictionary."""
        config = ModelConfig(
            model_name="test_model",
            temperature=0.2,
            max_tokens=100,
        )

        result = config.to_dict()
        expected = {
            "temperature": 0.2,
            "max_tokens": 100,
        }

        assert result == expected


class TestOpenAIConfig:
    """Test the OpenAIConfig dataclass."""

    def test_openai_config_creation(self):
        """Test creating an OpenAIConfig."""
        config = OpenAIConfig(
            model_name="gpt-4.1",
            temperature=0.1,
            max_tokens=150,
            json_mode=True,
            api_key_env="TEST_API_KEY",
        )

        assert config.model_name == "gpt-4.1"
        assert config.temperature == 0.1
        assert config.max_tokens == 150
        assert config.json_mode is True
        assert config.api_key_env == "TEST_API_KEY"

    @patch.dict("os.environ", {"TEST_API_KEY": "test_key_value"})
    def test_openai_config_to_dict(self):
        """Test converting OpenAIConfig to dictionary."""
        config = OpenAIConfig(
            model_name="gpt-4.1",
            temperature=0.1,
            max_tokens=150,
            json_mode=True,
            api_key_env="TEST_API_KEY",
        )

        result = config.to_dict()
        expected = {
            "temperature": 0.1,
            "max_tokens": 150,
            "json_mode": True,
            "api_key": "test_key_value",
        }

        assert result == expected


class TestOllamaConfig:
    """Test the OllamaConfig dataclass."""

    def test_ollama_config_creation(self):
        """Test creating an OllamaConfig."""
        config = OllamaConfig(
            model_name="qwen3:4b",
            temperature=0.1,
            max_tokens=150,
            host="http://localhost:11434",
            auto_pull=True,
        )

        assert config.model_name == "qwen3:4b"
        assert config.temperature == 0.1
        assert config.max_tokens == 150
        assert config.host == "http://localhost:11434"
        assert config.auto_pull is True

    def test_ollama_config_to_dict(self):
        """Test converting OllamaConfig to dictionary."""
        config = OllamaConfig(
            model_name="qwen3:4b",
            temperature=0.1,
            max_tokens=150,
            host="http://localhost:11434",
            auto_pull=True,
        )

        result = config.to_dict()
        expected = {
            "temperature": 0.1,
            "max_tokens": 150,
            "host": "http://localhost:11434",
        }

        assert result == expected


class TestExperimentConfig:
    """Test the ExperimentConfig dataclass."""

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig with default values."""
        config = ExperimentConfig()

        assert config.task_name == "sentiment_classification"
        assert config.data_split == "validation"
        assert config.n_samples == 50
        assert config.balanced_sampling is True
        assert config.random_seed == 42

    def test_experiment_config_custom(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            task_name="custom_task",
            data_split="test",
            n_samples=100,
            balanced_sampling=False,
            random_seed=123,
        )

        assert config.task_name == "custom_task"
        assert config.data_split == "test"
        assert config.n_samples == 100
        assert config.balanced_sampling is False
        assert config.random_seed == 123


class TestEvaluationConfig:
    """Test the EvaluationConfig dataclass."""

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig with default values."""
        config = EvaluationConfig()

        assert config.weighted_index_weights == {"accuracy": 0.7, "consistency": 0.3}
        assert "label_encodings" in config.custom_accuracy
        assert config.custom_accuracy["label_encodings"]["Very Negative"] == -3

    def test_evaluation_config_custom(self):
        """Test EvaluationConfig with custom values."""
        custom_weights = {"accuracy": 0.8, "consistency": 0.2}
        custom_accuracy = {"label_encodings": {"Positive": 1, "Negative": -1}}

        config = EvaluationConfig(
            weighted_index_weights=custom_weights,
            custom_accuracy=custom_accuracy,
        )

        assert config.weighted_index_weights == custom_weights
        assert config.custom_accuracy == custom_accuracy


class TestOutputConfig:
    """Test the OutputConfig dataclass."""

    def test_output_config_defaults(self):
        """Test OutputConfig with default values."""
        config = OutputConfig()

        assert config.results_dir == "results"
        assert config.log_level == "INFO"
        assert config.save_intermediate is True

    def test_output_config_custom(self):
        """Test OutputConfig with custom values."""
        config = OutputConfig(
            results_dir="custom_results",
            log_level="DEBUG",
            save_intermediate=False,
        )

        assert config.results_dir == "custom_results"
        assert config.log_level == "DEBUG"
        assert config.save_intermediate is False


class TestExperimentConfigLoader:
    """Test the ExperimentConfigLoader class."""

    def test_config_loader_with_valid_file(self):
        """Test loading configuration from a valid file."""
        config_data = {
            "models": {
                "openai": {
                    "model_name": "gpt-4.1",
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "json_mode": True,
                    "api_key_env": "OPENAI_API_KEY",
                },
                "ollama": {
                    "model_name": "qwen3:4b",
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "host": "http://localhost:11434",
                    "auto_pull": True,
                },
            },
            "experiment": {
                "task_name": "sentiment_classification",
                "data_split": "validation",
                "n_samples": 50,
                "balanced_sampling": True,
                "random_seed": 42,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loader = ExperimentConfigLoader(config_path)

            # Test OpenAI config
            openai_config = loader.get_openai_config()
            assert openai_config.model_name == "gpt-4.1"
            assert openai_config.temperature == 0.1

            # Test Ollama config
            ollama_config = loader.get_ollama_config()
            assert ollama_config.model_name == "qwen3:4b"
            assert ollama_config.temperature == 0.1

            # Test experiment config
            exp_config = loader.get_experiment_config()
            assert exp_config.task_name == "sentiment_classification"
            assert exp_config.n_samples == 50

        finally:
            Path(config_path).unlink()

    def test_config_loader_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ExperimentConfigLoader("nonexistent_config.json")

    def test_config_loader_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                ExperimentConfigLoader(config_path)
        finally:
            Path(config_path).unlink()

    def test_get_model_names(self):
        """Test getting model names."""
        config_data = {
            "models": {
                "openai": {"model_name": "gpt-4.1"},
                "ollama": {"model_name": "qwen3:4b"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loader = ExperimentConfigLoader(config_path)
            model_names = loader.get_model_names()

            assert model_names == {"openai": "gpt-4.1", "ollama": "qwen3:4b"}
        finally:
            Path(config_path).unlink()

    def test_update_config(self):
        """Test updating configuration."""
        config_data = {
            "models": {
                "openai": {"model_name": "gpt-4.1"},
                "ollama": {"model_name": "qwen3:4b"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loader = ExperimentConfigLoader(config_path)

            # Update configuration
            updates = {"models": {"openai": {"model_name": "gpt-4o-mini"}}}
            loader.update_config(updates)

            # Verify update
            openai_config = loader.get_openai_config()
            assert openai_config.model_name == "gpt-4o-mini"

            # Verify other values unchanged
            ollama_config = loader.get_ollama_config()
            assert ollama_config.model_name == "qwen3:4b"

        finally:
            Path(config_path).unlink()

    def test_save_config(self):
        """Test saving configuration."""
        config_data = {
            "models": {
                "openai": {"model_name": "gpt-4.1"},
                "ollama": {"model_name": "qwen3:4b"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loader = ExperimentConfigLoader(config_path)

            # Update and save
            updates = {"models": {"openai": {"model_name": "gpt-4o-mini"}}}
            loader.update_config(updates)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f2:
                output_path = f2.name

            try:
                loader.save_config(output_path)

                # Load saved config and verify
                with open(output_path, "r") as f:
                    saved_data = json.load(f)

                assert saved_data["models"]["openai"]["model_name"] == "gpt-4o-mini"
                assert saved_data["models"]["ollama"]["model_name"] == "qwen3:4b"

            finally:
                Path(output_path).unlink()

        finally:
            Path(config_path).unlink()


class TestLoadExperimentConfig:
    """Test the load_experiment_config convenience function."""

    def test_load_experiment_config_with_path(self):
        """Test loading config with explicit path."""
        config_data = {
            "models": {
                "openai": {"model_name": "gpt-4.1"},
                "ollama": {"model_name": "qwen3:4b"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loader = load_experiment_config(config_path)
            assert isinstance(loader, ExperimentConfigLoader)

            model_names = loader.get_model_names()
            assert model_names["openai"] == "gpt-4.1"
            assert model_names["ollama"] == "qwen3:4b"

        finally:
            Path(config_path).unlink()

    def test_load_experiment_config_default_path(self):
        """Test loading config with default path."""
        # This test assumes the default config exists
        try:
            loader = load_experiment_config()
            assert isinstance(loader, ExperimentConfigLoader)

            # Should be able to get configurations without error
            openai_config = loader.get_openai_config()
            ollama_config = loader.get_ollama_config()

            assert openai_config.model_name is not None
            assert ollama_config.model_name is not None

        except FileNotFoundError:
            # Skip test if default config doesn't exist
            pytest.skip("Default configuration file not found")


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_with_correct_model_names(self, test_config):
        """Test that configuration loads with correct model names."""
        openai_config = test_config.get_openai_config()
        ollama_config = test_config.get_ollama_config()

        # Test correct model names are loaded
        assert openai_config.model_name == "gpt-4.1"
        assert ollama_config.model_name == "qwen3:4b"

    def test_config_model_parameters(self, test_config):
        """Test that model parameters are correctly configured."""
        openai_config = test_config.get_openai_config()
        ollama_config = test_config.get_ollama_config()

        # Test OpenAI parameters
        assert openai_config.temperature == 0.1
        assert openai_config.max_tokens == 150
        assert openai_config.json_mode is True

        # Test Ollama parameters
        assert ollama_config.temperature == 0.1
        assert ollama_config.max_tokens == 150
        assert ollama_config.host == "http://localhost:11434"
        assert ollama_config.auto_pull is True

    def test_config_experiment_parameters(self, test_config):
        """Test that experiment parameters are correctly configured."""
        exp_config = test_config.get_experiment_config()

        assert exp_config.task_name == "sentiment_classification"
        assert exp_config.data_split == "validation"
        assert exp_config.balanced_sampling is True
        assert exp_config.random_seed == 42

    def test_config_evaluation_parameters(self, test_config):
        """Test that evaluation parameters are correctly configured."""
        eval_config = test_config.get_evaluation_config()

        assert eval_config.weighted_index_weights["accuracy"] == 0.7
        assert eval_config.weighted_index_weights["consistency"] == 0.3

        # Test custom accuracy encodings
        encodings = eval_config.custom_accuracy["label_encodings"]
        assert encodings["Very Negative"] == -3
        assert encodings["Negative"] == -2
        assert encodings["Neutral"] == 0
        assert encodings["Positive"] == 2
        assert encodings["Very Positive"] == 3
