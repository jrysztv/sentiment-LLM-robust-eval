"""
Configuration package for the Deep Learning Final Assignment.

This package provides configuration management functionality including:
- Enhanced configuration with flexible model selection
- Backward compatibility with existing configuration system
"""

# Re-export existing config functionality for backward compatibility
from ..config import (
    ModelConfig,
    OpenAIConfig,
    OllamaConfig,
    ExperimentConfig,
    EvaluationConfig,
    OutputConfig,
    ExperimentConfigLoader,
    load_experiment_config,
)

# Export flexible configuration functionality
from .flexible_config import (
    FlexibleModelConfig,
    FlexibleConfigLoader,
    parse_cli_models,
    create_config_from_cli,
)

__all__ = [
    # Existing config classes
    "ModelConfig",
    "OpenAIConfig",
    "OllamaConfig",
    "ExperimentConfig",
    "EvaluationConfig",
    "OutputConfig",
    "ExperimentConfigLoader",
    "load_experiment_config",
    # Flexible config classes
    "FlexibleModelConfig",
    "FlexibleConfigLoader",
    "parse_cli_models",
    "create_config_from_cli",
    # Backward compatibility aliases
    "ConfigLoader",
]

# Backward compatibility aliases
ConfigLoader = FlexibleConfigLoader
