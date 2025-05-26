"""
Configuration management for the Deep Learning Final Assignment.

This module handles loading and validating experiment configurations
from JSON files, providing a centralized way to manage model settings,
experiment parameters, and evaluation criteria.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_name: str
    temperature: float = 0.1
    max_tokens: int = 150

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass
class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""

    json_mode: bool = True
    api_key_env: str = "OPENAI_API_KEY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "json_mode": self.json_mode,
                "api_key": os.getenv(self.api_key_env),
            }
        )
        return base_dict


@dataclass
class OllamaConfig(ModelConfig):
    """Configuration for Ollama models."""

    host: str = "http://localhost:11434"
    auto_pull: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model initialization."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "host": self.host,
            }
        )
        return base_dict


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    task_name: str = "sentiment_classification"
    data_split: str = "validation"
    n_samples: int = 50
    balanced_sampling: bool = True
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    weighted_index_weights: Dict[str, float] = None
    custom_accuracy: Dict[str, Any] = None

    def __post_init__(self):
        if self.weighted_index_weights is None:
            self.weighted_index_weights = {"accuracy": 0.7, "consistency": 0.3}

        if self.custom_accuracy is None:
            self.custom_accuracy = {
                "label_encodings": {
                    "Very Negative": -3,
                    "Negative": -2,
                    "Neutral": 0,
                    "Positive": 2,
                    "Very Positive": 3,
                }
            }


@dataclass
class OutputConfig:
    """Configuration for output and logging."""

    results_dir: str = "results"
    log_level: str = "INFO"
    save_intermediate: bool = True


class ExperimentConfigLoader:
    """Loads and manages experiment configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to config/experiment_config.json
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "experiment_config.json"

        self.config_path = Path(config_path)
        self._config_data = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self._config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def get_openai_config(self) -> OpenAIConfig:
        """Get OpenAI model configuration."""
        openai_data = self._config_data.get("models", {}).get("openai", {})
        return OpenAIConfig(
            model_name=openai_data.get("model_name", "gpt-4o"),
            temperature=openai_data.get("temperature", 0.1),
            max_tokens=openai_data.get("max_tokens", 150),
            json_mode=openai_data.get("json_mode", True),
            api_key_env=openai_data.get("api_key_env", "OPENAI_API_KEY"),
        )

    def get_ollama_config(self) -> OllamaConfig:
        """Get Ollama model configuration."""
        ollama_data = self._config_data.get("models", {}).get("ollama", {})
        return OllamaConfig(
            model_name=ollama_data.get("model_name", "qwen2.5:7b"),
            temperature=ollama_data.get("temperature", 0.1),
            max_tokens=ollama_data.get("max_tokens", 150),
            host=ollama_data.get("host", "http://localhost:11434"),
            auto_pull=ollama_data.get("auto_pull", True),
        )

    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        exp_data = self._config_data.get("experiment", {})
        return ExperimentConfig(
            task_name=exp_data.get("task_name", "sentiment_classification"),
            data_split=exp_data.get("data_split", "validation"),
            n_samples=exp_data.get("n_samples", 50),
            balanced_sampling=exp_data.get("balanced_sampling", True),
            random_seed=exp_data.get("random_seed", 42),
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        eval_data = self._config_data.get("evaluation", {})
        return EvaluationConfig(
            weighted_index_weights=eval_data.get("weighted_index_weights"),
            custom_accuracy=eval_data.get("custom_accuracy"),
        )

    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        output_data = self._config_data.get("output", {})
        return OutputConfig(
            results_dir=output_data.get("results_dir", "results"),
            log_level=output_data.get("log_level", "INFO"),
            save_intermediate=output_data.get("save_intermediate", True),
        )

    def get_model_names(self) -> Dict[str, str]:
        """Get model names for both providers."""
        return {
            "openai": self.get_openai_config().model_name,
            "ollama": self.get_ollama_config().model_name,
        }

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """

        def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict

        deep_update(self._config_data, updates)

    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        save_path = Path(output_path) if output_path else self.config_path

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self._config_data, f, indent=2)


def load_experiment_config(config_path: Optional[str] = None) -> ExperimentConfigLoader:
    """
    Convenience function to load experiment configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration manager
    """
    return ExperimentConfigLoader(config_path)
