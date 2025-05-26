"""
Enhanced configuration management for Phase 1.5.

This module implements flexible model selection and runtime configuration
that supports:
- enabled_models parameter for runtime model selection
- CLI model overrides (--models gpt-4.1,gpt-4o-mini)
- Dynamic provider loading based on enabled models
- Backward compatibility with existing configurations
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..config import load_experiment_config


@dataclass
class FlexibleModelConfig:
    """Flexible model configuration with runtime selection."""

    enabled_models: List[str]
    openai_models: Dict[str, Dict[str, Any]]
    ollama_models: Dict[str, Dict[str, Any]]

    def get_enabled_providers(self) -> List[str]:
        """Get list of providers needed for enabled models."""
        providers = set()

        for model_name in self.enabled_models:
            if model_name in self.openai_models:
                providers.add("openai")
            elif model_name in self.ollama_models:
                providers.add("ollama")

        return list(providers)

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        if model_name in self.openai_models:
            return {"provider": "openai", "config": self.openai_models[model_name]}
        elif model_name in self.ollama_models:
            return {"provider": "ollama", "config": self.ollama_models[model_name]}

        return None

    def validate_enabled_models(self) -> bool:
        """Validate that all enabled models have configurations."""
        for model_name in self.enabled_models:
            if self.get_model_config(model_name) is None:
                raise ValueError(
                    f"No configuration found for enabled model: {model_name}"
                )

        return True


class FlexibleConfigLoader:
    """Flexible configuration loader with runtime model selection."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced configuration loader.

        Args:
            config_path: Path to configuration file
        """
        self.base_config_loader = load_experiment_config(config_path)
        self._enhanced_model_config = None

    def load_enhanced_model_config(
        self, cli_models: Optional[List[str]] = None
    ) -> FlexibleModelConfig:
        """
        Load flexible model configuration with CLI overrides.

        Args:
            cli_models: Models specified via CLI (--models param)

        Returns:
            FlexibleModelConfig with enabled models
        """
        # Get base configuration
        config_data = self.base_config_loader._config_data

        # Extract model configurations
        models_config = config_data.get("models", {})

        # Build OpenAI model options
        openai_models = {}
        if "openai" in models_config:
            openai_config = models_config["openai"]
            model_options = openai_config.get("model_options", {})

            # If no model_options, use the direct configuration (backward compatibility)
            if not model_options and "model_name" in openai_config:
                model_name = openai_config["model_name"]
                openai_models[model_name] = {
                    k: v for k, v in openai_config.items() if k != "model_name"
                }
            else:
                openai_models = model_options

        # Build Ollama model options
        ollama_models = {}
        if "ollama" in models_config:
            ollama_config = models_config["ollama"]
            model_options = ollama_config.get("model_options", {})

            # If no model_options, use the direct configuration (backward compatibility)
            if not model_options and "model_name" in ollama_config:
                model_name = ollama_config["model_name"]
                ollama_models[model_name] = {
                    k: v for k, v in ollama_config.items() if k != "model_name"
                }
            else:
                ollama_models = model_options

        # Determine enabled models
        if cli_models:
            # CLI override takes precedence
            enabled_models = cli_models
        elif "enabled_models" in models_config:
            # Use configuration file setting
            enabled_models = models_config["enabled_models"]
        else:
            # Backward compatibility: enable all configured models
            enabled_models = list(openai_models.keys()) + list(ollama_models.keys())

        # Create flexible config
        enhanced_config = FlexibleModelConfig(
            enabled_models=enabled_models,
            openai_models=openai_models,
            ollama_models=ollama_models,
        )

        # Validate configuration
        enhanced_config.validate_enabled_models()

        self._enhanced_model_config = enhanced_config
        return enhanced_config

    def get_enabled_models(self) -> List[str]:
        """Get list of enabled model names."""
        if self._enhanced_model_config is None:
            self.load_enhanced_model_config()

        return self._enhanced_model_config.enabled_models

    def get_enabled_providers(self) -> List[str]:
        """Get list of providers needed for enabled models."""
        if self._enhanced_model_config is None:
            self.load_enhanced_model_config()

        return self._enhanced_model_config.get_enabled_providers()

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        if self._enhanced_model_config is None:
            self.load_enhanced_model_config()

        return self._enhanced_model_config.get_model_config(model_name)

    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled."""
        return model_name in self.get_enabled_models()

    def get_experiment_config(self):
        """Get experiment configuration (delegates to base loader)."""
        return self.base_config_loader.get_experiment_config()

    def get_evaluation_config(self):
        """Get evaluation configuration (delegates to base loader)."""
        return self.base_config_loader.get_evaluation_config()

    def get_output_config(self):
        """Get output configuration (delegates to base loader)."""
        return self.base_config_loader.get_output_config()


def parse_cli_models(models_str: str) -> List[str]:
    """
    Parse CLI models string into list of model names.

    Args:
        models_str: Comma-separated model names (e.g., "gpt-4.1,gpt-4o-mini")

    Returns:
        List of model names
    """
    return [model.strip() for model in models_str.split(",") if model.strip()]


def create_config_from_cli(
    config_path: Optional[str] = None, models_str: Optional[str] = None
) -> FlexibleConfigLoader:
    """
    Create flexible configuration loader with CLI model selection.

    Args:
        config_path: Path to configuration file
        models_str: Comma-separated model names from CLI

    Returns:
        FlexibleConfigLoader configured with CLI overrides
    """
    loader = FlexibleConfigLoader(config_path)

    # Parse CLI models if provided
    cli_models = None
    if models_str:
        cli_models = parse_cli_models(models_str)

    # Load enhanced configuration with CLI overrides
    loader.load_enhanced_model_config(cli_models)

    return loader
