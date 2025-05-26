#!/usr/bin/env python3
"""
Context-Enhanced Prompt Execution Script

Runs the specific context-enhanced variants identified in Phase 2:
- Reads best variants from context_enhanced_prompts file
- Executes only the 4 enhanced variants (2 models × 2 context positions)
- Uses existing evaluation methods and data persistence
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_learning_final_assignment.core.models import (
    AsyncOpenAIModel,
    AsyncOllamaModel,
)
from deep_learning_final_assignment.core.prompts.template import PromptVariant
from deep_learning_final_assignment.core.data import SST5Loader
from deep_learning_final_assignment.core.evaluation import RobustnessEvaluator
from deep_learning_final_assignment.core.config_pkg import (
    FlexibleConfigLoader,
    parse_cli_models,
)
from deep_learning_final_assignment.core.utils import (
    setup_unicode_safe_logging,
    get_safe_logger,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup Unicode-safe logging configuration."""
    setup_unicode_safe_logging(
        log_level=log_level, log_file="context_enhanced_execution.log"
    )
    return get_safe_logger(__name__)


def load_context_enhanced_prompts(results_dir: str = "results") -> Dict[str, Any]:
    """Load the most recent context_enhanced_prompts file."""
    results_path = Path(results_dir)

    # Find the most recent context_enhanced_prompts file
    pattern = "context_enhanced_prompts_*.json"
    files = list(results_path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No context_enhanced_prompts files found in {results_dir}"
        )

    # Sort by timestamp and get the most recent
    latest_file = sorted(files, key=lambda x: x.name)[-1]

    with open(latest_file, "r") as f:
        data = json.load(f)

    return data, latest_file


async def initialize_models(config_loader: FlexibleConfigLoader) -> Dict[str, Any]:
    """Initialize async models based on enabled models in configuration."""
    logger = logging.getLogger(__name__)

    models = {}
    enabled_models = config_loader.get_enabled_models()

    for model_name in enabled_models:
        model_config = config_loader.get_model_config(model_name)
        if not model_config:
            logger.warning(f"⚠️ No configuration found for model: {model_name}")
            continue

        provider = model_config["provider"]
        config_dict = model_config["config"]

        if provider == "openai":
            model = AsyncOpenAIModel(
                model_name=model_name,
                api_key=os.getenv(config_dict["api_key_env"]),
                json_mode=config_dict.get("json_mode", True),
            )
        elif provider == "ollama":
            model = AsyncOllamaModel(
                model_name=model_name,
                host=config_dict.get("host", "http://localhost:11434"),
                auto_pull=config_dict.get("auto_pull", True),
            )
        else:
            logger.warning(f"⚠️ Unknown provider for model {model_name}: {provider}")
            continue

        models[model_name] = model
        logger.info(f"✅ Initialized {model_name} ({provider})")

    return models


def create_enhanced_prompt_variant(
    variant_id: str, template: str, context_position: str, base_variant: str
) -> PromptVariant:
    """Create a PromptVariant object from enhanced prompt data."""
    # Create dimensions based on base variant and context enhancement
    dimensions = {
        "formality": "unknown",  # Will be inherited from base
        "phrasing": "unknown",  # Will be inherited from base
        "order": "unknown",  # Will be inherited from base
        "synonyms": "unknown",  # Will be inherited from base
        "context_position": context_position,
        "base_variant": base_variant,
    }

    # Create a descriptive name
    name = f"Context Enhanced {base_variant} ({context_position})"

    return PromptVariant(
        id=variant_id,
        name=name,
        template=template,
        dimensions=dimensions,
        description=f"Context-enhanced variant with {context_position} positioning based on {base_variant}",
    )


async def run_single_enhanced_combination(
    model_name: str,
    model: Any,
    variant: PromptVariant,
    test_samples: List[Any],
    config_loader: Any,
    evaluator: RobustnessEvaluator,
    combination_id: str,
) -> tuple:
    """Run a single enhanced model-prompt combination asynchronously."""
    logger = logging.getLogger(__name__)

    try:
        # Format prompts for all test samples
        formatted_prompts = [
            variant.format(input_text=sample.text) for sample in test_samples
        ]

        # Generate responses using model configuration
        model_config = config_loader.get_model_config(model_name)
        if model_config is None:
            raise ValueError(f"No configuration found for model: {model_name}")

        config_dict = model_config["config"]

        logger.info(
            f"🚀 Running {combination_id} with {len(formatted_prompts)} samples..."
        )

        responses = await model.batch_generate(
            formatted_prompts,
            temperature=config_dict.get("temperature", 0.1),
            max_tokens=config_dict.get("max_tokens", 150),
        )

        # Evaluate this combination
        result = evaluator.evaluate_single_combination(
            responses, test_samples, variant.id
        )

        logger.info(
            f"✅ {combination_id}: "
            f"Accuracy={result.custom_accuracy:.3f}, "
            f"Weighted={result.weighted_index:.3f}"
        )

        return (model_name, variant.id, result)

    except Exception as e:
        logger.error(f"❌ Failed {combination_id}: {e}")
        return (model_name, variant.id, None)


async def run_context_enhanced_experiment(
    enhanced_prompts_data: Dict[str, Any],
    config_loader: FlexibleConfigLoader,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Run context-enhanced experiment for all enhanced variants."""
    logger = logging.getLogger(__name__)

    # Initialize components
    logger.info("🚀 Starting context-enhanced experiment execution")

    # Get configurations
    exp_config = config_loader.get_experiment_config()

    # Load the same test samples used in baseline
    logger.info("📊 Loading SST-5 dataset...")
    data_loader = SST5Loader()
    test_samples = data_loader.get_sample_subset(
        split=exp_config.data_split,
        n_samples=n_samples,
        balanced=exp_config.balanced_sampling,
        random_seed=exp_config.random_seed,
    )
    logger.info(
        f"✅ Loaded {len(test_samples)} balanced samples from {exp_config.data_split} split"
    )

    # Initialize models
    logger.info("🤖 Initializing async models...")
    models = await initialize_models(config_loader)
    logger.info(f"✅ Initialized {len(models)} async models: {list(models.keys())}")

    # Initialize evaluator
    evaluator = RobustnessEvaluator()

    # Create enhanced variants
    enhanced_variants = []
    context_variants_data = enhanced_prompts_data["context_enhanced_variants"]

    for model_name, model_data in context_variants_data.items():
        if model_name not in models:
            logger.warning(f"⚠️ Model {model_name} not available, skipping...")
            continue

        # Create prefix variant
        prefix_data = model_data["prefix_context"]
        prefix_variant = create_enhanced_prompt_variant(
            variant_id=prefix_data["variant_id"],
            template=prefix_data["full_template"],
            context_position=prefix_data["context_position"],
            base_variant=prefix_data["base_variant"],
        )
        enhanced_variants.append((model_name, prefix_variant))

        # Create suffix variant
        suffix_data = model_data["suffix_context"]
        suffix_variant = create_enhanced_prompt_variant(
            variant_id=suffix_data["variant_id"],
            template=suffix_data["full_template"],
            context_position=suffix_data["context_position"],
            base_variant=suffix_data["base_variant"],
        )
        enhanced_variants.append((model_name, suffix_variant))

    # Create execution tasks
    logger.info(
        f"🔬 Creating async tasks for {len(enhanced_variants)} enhanced combinations..."
    )
    tasks = []

    for model_name, variant in enhanced_variants:
        model = models[model_name]
        combination_id = f"{model_name} + {variant.id}"
        task = run_single_enhanced_combination(
            model_name,
            model,
            variant,
            test_samples,
            config_loader,
            evaluator,
            combination_id,
        )
        tasks.append(task)

    logger.info(
        f"🚀 Running {len(enhanced_variants)} enhanced combinations concurrently..."
    )

    # Run all combinations concurrently
    start_time = datetime.now()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.now()

    execution_time = (end_time - start_time).total_seconds()
    logger.info(
        f"⚡ Completed all enhanced combinations in {execution_time:.2f} seconds"
    )

    # Organize results by model
    all_results = {}
    successful_combinations = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            continue

        model_name, variant_id, evaluation_result = result

        if evaluation_result is not None:
            if model_name not in all_results:
                all_results[model_name] = {}
            all_results[model_name][variant_id] = evaluation_result
            successful_combinations += 1

    logger.info(
        f"✅ Successfully completed {successful_combinations}/{len(enhanced_variants)} enhanced combinations"
    )

    # Close async models
    logger.info("🔄 Closing async model connections...")
    for model in models.values():
        if hasattr(model, "close"):
            await model.close()

    return all_results


def save_enhanced_results(
    results: Dict[str, Any],
    enhanced_prompts_data: Dict[str, Any],
    output_dir: str = "results",
) -> None:
    """Save context-enhanced experiment results to files."""
    logger = logging.getLogger(__name__)

    # Create results directory
    results_dir = Path(output_dir)
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detailed_file = results_dir / f"context_enhanced_results_detailed_{timestamp}.json"

    # Convert results to serializable format
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {}
        for variant_id, result in model_results.items():
            serializable_results[model_name][variant_id] = {
                "model_name": result.model_name,
                "prompt_variant_id": result.prompt_variant_id,
                "custom_accuracy": result.custom_accuracy,
                "consistency_score": result.consistency_score,
                "weighted_index": result.weighted_index,
                "error_breakdown": result.error_breakdown,
                "metadata": result.metadata,
                "predictions": result.predictions,
                "true_labels": result.true_labels,
            }

    with open(detailed_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"💾 Enhanced detailed results saved to {detailed_file}")

    # Save summary results with baseline comparison
    summary_file = results_dir / f"context_enhanced_results_summary_{timestamp}.json"
    summary_data = {
        "experiment_metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "context_enhanced_execution",
            "total_models": len(results),
            "variants_per_model": 2,  # prefix + suffix
        },
        "baseline_reference": enhanced_prompts_data["best_baseline_combinations"],
        "enhanced_results": {},
    }

    for model_name, model_results in results.items():
        if not model_results:
            continue

        evaluator = RobustnessEvaluator()
        best_variant_id, best_result = evaluator.get_best_combination(model_results)

        # Get baseline comparison data
        baseline_data = enhanced_prompts_data["best_baseline_combinations"][model_name]

        summary_data["enhanced_results"][model_name] = {
            "baseline_performance": {
                "variant_id": baseline_data["variant_id"],
                "custom_accuracy": baseline_data["custom_accuracy"],
                "consistency_score": baseline_data["consistency_score"],
                "weighted_index": baseline_data["weighted_index"],
            },
            "enhanced_performance": {
                "best_variant_id": best_variant_id,
                "custom_accuracy": best_result.custom_accuracy,
                "consistency_score": best_result.consistency_score,
                "weighted_index": best_result.weighted_index,
            },
            "improvement": {
                "accuracy_improvement": best_result.custom_accuracy
                - baseline_data["custom_accuracy"],
                "consistency_improvement": best_result.consistency_score
                - baseline_data["consistency_score"],
                "weighted_index_improvement": best_result.weighted_index
                - baseline_data["weighted_index"],
            },
            "all_enhanced_variants": {
                variant_id: {
                    "custom_accuracy": result.custom_accuracy,
                    "consistency_score": result.consistency_score,
                    "weighted_index": result.weighted_index,
                    "context_position": result.metadata.get(
                        "context_position", "unknown"
                    ),
                }
                for variant_id, result in model_results.items()
            },
        }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"📊 Enhanced summary results saved to {summary_file}")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run context-enhanced prompt execution experiment"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file (default: config/experiment_config.json)",
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to use (e.g., 'gpt-4.1,gpt-4o-mini')",
    )

    # Override options
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of test samples (default: 50)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory to read context-enhanced prompts from",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load context-enhanced prompts
    try:
        enhanced_prompts_data, prompts_file = load_context_enhanced_prompts(
            args.results_dir
        )
        logger = setup_logging(args.log_level)
        logger.info(f"✅ Loaded context-enhanced prompts from {prompts_file}")
    except Exception as e:
        print(f"❌ Failed to load context-enhanced prompts: {e}")
        return 1

    # Load configuration
    try:
        config_loader = FlexibleConfigLoader(args.config)

        # Parse CLI models if provided
        cli_models = None
        if args.models:
            cli_models = parse_cli_models(args.models)

        # Load enhanced model configuration
        config_loader.load_enhanced_model_config(cli_models)

        logger.info(f"✅ Configuration loaded")

        if cli_models:
            logger.info(f"🔧 CLI model override: {cli_models}")

        enabled_models = config_loader.get_enabled_models()
        logger.info(f"🎯 Enabled models: {enabled_models}")

    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1

    # Validate that enabled models match those in enhanced prompts
    enhanced_models = set(enhanced_prompts_data["context_enhanced_variants"].keys())
    enabled_models_set = set(enabled_models)

    if not enabled_models_set.intersection(enhanced_models):
        logger.error(
            f"❌ No overlap between enabled models {enabled_models} and enhanced prompt models {enhanced_models}"
        )
        return 1

    try:
        # Run enhanced experiment
        logger.info(f"🎯 Target models: {list(enhanced_models & enabled_models_set)}")
        logger.info(f"📊 Test samples: {args.n_samples}")

        results = await run_context_enhanced_experiment(
            enhanced_prompts_data, config_loader, args.n_samples
        )

        # Save results
        save_enhanced_results(results, enhanced_prompts_data, args.results_dir)

        # Print summary
        logger.info("🎉 Context-enhanced experiment completed successfully!")

        total_combinations = sum(
            len(model_results) for model_results in results.values()
        )
        logger.info(f"📊 Total enhanced combinations tested: {total_combinations}")

        # Display results summary
        for model_name, model_results in results.items():
            if model_results:
                evaluator = RobustnessEvaluator()
                best_variant_id, best_result = evaluator.get_best_combination(
                    model_results
                )

                baseline_accuracy = enhanced_prompts_data["best_baseline_combinations"][
                    model_name
                ]["custom_accuracy"]
                improvement = best_result.custom_accuracy - baseline_accuracy

                logger.info(
                    f"🏆 Best {model_name}: {best_variant_id} "
                    f"(Accuracy: {best_result.custom_accuracy:.3f}, "
                    f"Improvement: {improvement:+.3f})"
                )

    except Exception as e:
        logger.error(f"💥 Context-enhanced experiment failed: {e}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
