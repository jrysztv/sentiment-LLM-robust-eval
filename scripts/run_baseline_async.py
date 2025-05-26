#!/usr/bin/env python3
"""
Async baseline robustness testing script with rate limiting.

Runs all combinations (16 prompt variants × N models) asynchronously and saves results.
This implements Phase 1.5 with flexible model selection support.
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
from deep_learning_final_assignment.core.prompts import (
    SentimentPrompts,
    PromptPerturbator,
)
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
from deep_learning_final_assignment.core.context_enhancement import (
    Phase2ContextEnhancement,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup Unicode-safe logging configuration."""
    # Use our Unicode-safe logging setup
    setup_unicode_safe_logging(
        log_level=log_level, log_file="baseline_async_experiment.log"
    )
    return get_safe_logger(__name__)


async def initialize_models(config_loader: FlexibleConfigLoader) -> Dict[str, Any]:
    """Initialize async models based on enabled models in configuration."""
    models = {}
    logger = logging.getLogger(__name__)

    # Get enabled models
    enabled_models = config_loader.get_enabled_models()
    logger.info(f"🎯 Enabled models: {enabled_models}")

    for model_name in enabled_models:
        model_config = config_loader.get_model_config(model_name)

        if model_config is None:
            logger.warning(f"❌ No configuration found for model: {model_name}")
            continue

        provider = model_config["provider"]
        config_dict = model_config["config"]

        try:
            if provider == "openai":
                model = AsyncOpenAIModel(model_name=model_name, **config_dict)
                if model.is_available:
                    models[model_name] = model
                    logger.info(
                        f"✅ OpenAI model '{model_name}' initialized successfully"
                    )
                else:
                    logger.warning(f"❌ OpenAI model '{model_name}' not available")

            elif provider == "ollama":
                model = AsyncOllamaModel(model_name=model_name, **config_dict)
                if model.is_available:
                    models[model_name] = model
                    logger.info(
                        f"✅ Ollama model '{model_name}' initialized successfully"
                    )
                else:
                    logger.warning(f"❌ Ollama model '{model_name}' not available")
                    # Try to pull the model if auto_pull is enabled
                    if config_dict.get("auto_pull", False):
                        logger.info(
                            f"🔄 Attempting to pull Ollama model '{model_name}'..."
                        )
                        if await model.pull_model():
                            if model.is_available:
                                models[model_name] = model
                                logger.info(
                                    f"✅ Ollama model '{model_name}' pulled and initialized"
                                )
                            else:
                                logger.error(
                                    f"❌ Failed to initialize Ollama model after pull"
                                )
                        else:
                            logger.error(
                                f"❌ Failed to pull Ollama model '{model_name}'"
                            )
                    else:
                        logger.info(
                            "Auto-pull disabled. Please manually pull the model or enable auto_pull in config."
                        )
            else:
                logger.error(
                    f"❌ Unknown provider '{provider}' for model '{model_name}'"
                )

        except Exception as e:
            logger.error(f"❌ Failed to initialize model '{model_name}': {e}")

    if not models:
        raise RuntimeError("No models available for testing")

    logger.info(
        f"✅ Successfully initialized {len(models)} models: {list(models.keys())}"
    )
    return models


async def run_single_combination(
    model_name: str,
    model: Any,
    variant: Any,
    test_samples: List[Any],
    config_loader: Any,
    evaluator: RobustnessEvaluator,
    combination_id: str,
) -> tuple:
    """Run a single model-prompt combination asynchronously."""
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


async def run_baseline_experiment(
    config_loader: FlexibleConfigLoader,
) -> Dict[str, Any]:
    """Run the complete baseline robustness experiment asynchronously."""
    logger = logging.getLogger(__name__)

    # Initialize components
    logger.info("🚀 Starting async baseline robustness experiment")

    # Get configurations
    exp_config = config_loader.get_experiment_config()
    eval_config = config_loader.get_evaluation_config()
    output_config = config_loader.get_output_config()

    # Load data
    logger.info("📊 Loading SST-5 dataset...")
    data_loader = SST5Loader()
    test_samples = data_loader.get_sample_subset(
        split=exp_config.data_split,
        n_samples=exp_config.n_samples,
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

    # Setup prompts
    logger.info("📝 Setting up prompt variants...")
    sentiment_prompts = SentimentPrompts(exp_config.task_name)
    perturbator = PromptPerturbator(sentiment_prompts)
    all_variants = perturbator.get_baseline_variants()
    logger.info(f"✅ Loaded {len(all_variants)} prompt variants")

    # Initialize evaluator
    evaluator = RobustnessEvaluator()

    # Create all combination tasks
    logger.info("🔬 Creating async tasks for all model-prompt combinations...")
    tasks = []
    total_combinations = len(models) * len(all_variants)

    for model_name, model in models.items():
        for variant in all_variants:
            combination_id = f"{model_name} + {variant.id}"
            task = run_single_combination(
                model_name,
                model,
                variant,
                test_samples,
                config_loader,
                evaluator,
                combination_id,
            )
            tasks.append(task)

    logger.info(f"🚀 Running {total_combinations} combinations concurrently...")

    # Run all combinations concurrently
    start_time = datetime.now()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.now()

    execution_time = (end_time - start_time).total_seconds()
    logger.info(f"⚡ Completed all combinations in {execution_time:.2f} seconds")

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
        f"✅ Successfully completed {successful_combinations}/{total_combinations} combinations"
    )

    # Calculate cross-variant consistency for each model
    logger.info("📈 Calculating cross-variant consistency...")
    final_results = {}

    for model_name, model_results in all_results.items():
        if not model_results:
            continue

        # Prepare data for cross-variant evaluation
        combination_responses = {}
        variant_dimensions = {}

        for variant_id, result in model_results.items():
            # Reconstruct responses from results (simplified)
            variant = sentiment_prompts.get_variant(variant_id)
            variant_dimensions[variant_id] = variant.dimensions

            # Store predictions as mock responses for consistency calculation
            combination_responses[variant_id] = result.predictions

        # Calculate consistency across variants for this model
        if len(combination_responses) > 1:
            variant_predictions = {
                vid: preds for vid, preds in combination_responses.items()
            }
            overall_consistency = evaluator.consistency_metric.calculate_consistency(
                variant_predictions
            )
            dimensional_consistency = (
                evaluator.consistency_metric.calculate_dimensional_consistency(
                    variant_predictions, variant_dimensions
                )
            )

            # Update results with cross-variant consistency
            for variant_id, result in model_results.items():
                result.consistency_score = overall_consistency
                result.weighted_index = (
                    eval_config.weighted_index_weights["accuracy"]
                    * result.custom_accuracy
                    + eval_config.weighted_index_weights["consistency"]
                    * overall_consistency
                )
                result.metadata["dimensional_consistency"] = dimensional_consistency
                result.metadata["overall_consistency"] = overall_consistency

        final_results[model_name] = model_results

    # Close async models
    logger.info("🔄 Closing async model connections...")
    for model in models.values():
        if hasattr(model, "close"):
            await model.close()

    return final_results


def save_results(results: Dict[str, Any], config_loader: Any) -> None:
    """Save experiment results to files."""
    logger = logging.getLogger(__name__)

    # Get output configuration
    output_config = config_loader.get_output_config()

    # Create results directory
    results_dir = Path(output_config.results_dir)
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detailed_file = results_dir / f"baseline_async_results_detailed_{timestamp}.json"

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

    logger.info(f"💾 Detailed async results saved to {detailed_file}")

    # Save summary results
    summary_file = results_dir / f"baseline_async_results_summary_{timestamp}.json"
    summary_data = {}

    for model_name, model_results in results.items():
        if not model_results:
            continue

        evaluator = RobustnessEvaluator()
        summary = evaluator.get_performance_summary(model_results)
        best_variant_id, best_result = evaluator.get_best_combination(model_results)

        summary_data[model_name] = {
            "performance_summary": summary,
            "best_combination": {
                "variant_id": best_variant_id,
                "custom_accuracy": best_result.custom_accuracy,
                "consistency_score": best_result.consistency_score,
                "weighted_index": best_result.weighted_index,
            },
            "total_variants_tested": len(model_results),
        }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"📊 Summary async results saved to {summary_file}")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run async baseline robustness testing experiment"
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

    # Phase 2 context enhancement
    parser.add_argument(
        "--run-phase2",
        action="store_true",
        help="Run Phase 2 context enhancement after baseline experiment",
    )

    # Override options (optional - config file takes precedence)
    parser.add_argument(
        "--openai-model", default=None, help="Override OpenAI model name from config"
    )
    parser.add_argument(
        "--ollama-model", default=None, help="Override Ollama model name from config"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of samples from config",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level from config",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        # Create flexible configuration loader
        config_loader = FlexibleConfigLoader(args.config)

        # Parse CLI models if provided
        cli_models = None
        if args.models:
            cli_models = parse_cli_models(args.models)

        # Load enhanced model configuration
        config_loader.load_enhanced_model_config(cli_models)

        # Setup logging
        logger = setup_logging(
            args.log_level or config_loader.get_output_config().log_level
        )
        logger.info(f"✅ Configuration loaded")

        if cli_models:
            logger.info(f"🔧 CLI model override: {cli_models}")

        enabled_models = config_loader.get_enabled_models()
        logger.info(f"🎯 Enabled models: {enabled_models}")

        # Apply legacy command line overrides if provided (deprecated)
        if args.openai_model:
            logger.warning("⚠️  --openai-model is deprecated, use --models instead")

        if args.ollama_model:
            logger.warning("⚠️  --ollama-model is deprecated, use --models instead")

        # Apply sample count override
        if args.n_samples:
            # This is a bit tricky with the flexible config, so we'll modify the base config
            base_config = config_loader.base_config_loader
            base_config.update_config({"experiment": {"n_samples": args.n_samples}})
            logger.info(f"🔧 Overriding sample count to: {args.n_samples}")

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1

    # Display current configuration
    exp_config = config_loader.get_experiment_config()
    enabled_models = config_loader.get_enabled_models()
    logger.info(f"🤖 Enabled models: {enabled_models}")
    logger.info(
        f"📊 Dataset: {exp_config.data_split} split, {exp_config.n_samples} samples"
    )

    try:
        # Run experiment
        results = await run_baseline_experiment(config_loader)

        # Save results
        save_results(results, config_loader)

        # Print summary
        logger.info("🎉 Async baseline experiment completed successfully!")

        total_combinations = sum(
            len(model_results) for model_results in results.values()
        )
        logger.info(f"📊 Total combinations tested: {total_combinations}")

        # Display best combinations per model (Phase 2 requirement)
        if results:
            evaluator = RobustnessEvaluator()
            best_per_model = evaluator.get_best_combinations_per_model(results)

            for model_name, (best_variant_id, best_result) in best_per_model.items():
                logger.info(
                    f"🏆 Best {model_name}: {best_variant_id} "
                    f"(Weighted Index: {best_result.weighted_index:.3f})"
                )

        # Run Phase 2 context enhancement if requested
        if args.run_phase2:
            logger.info("🚀 Starting Phase 2 context enhancement...")

            # Get test samples used in baseline
            exp_config = config_loader.get_experiment_config()
            data_loader = SST5Loader()
            test_samples = data_loader.get_sample_subset(
                split=exp_config.data_split,
                n_samples=exp_config.n_samples,
                balanced=exp_config.balanced_sampling,
                random_seed=exp_config.random_seed,
            )

            # Initialize Phase 2 system
            sentiment_prompts = SentimentPrompts(exp_config.task_name)
            phase2_system = Phase2ContextEnhancement(data_loader, sentiment_prompts)

            # Run context enhancement
            phase2_results = phase2_system.run_context_enhancement(
                baseline_results=results,
                test_samples=test_samples,
                n_context_per_label=3,
                random_seed=exp_config.random_seed,
            )

            # Save Phase 2 results
            saved_files = phase2_system.save_phase2_results(
                phase2_results, output_dir=config_loader.get_output_config().results_dir
            )

            logger.info("✅ Phase 2 context enhancement completed!")
            logger.info(f"📁 Phase 2 files saved: {list(saved_files.keys())}")

            # Display Phase 2 summary
            for model_name, data in phase2_results["enhanced_variants"].items():
                base_info = data["base_variant"]
                logger.info(
                    f"🔧 {model_name} enhanced from {base_info['variant_id']} "
                    f"(baseline accuracy: {base_info['custom_accuracy']:.3f})"
                )

    except Exception as e:
        logger.error(f"💥 Async experiment failed: {e}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
