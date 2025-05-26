"""
Data persistence manager for Phase 1.5 enhanced metrics.

This module implements the DataPersistenceManager that handles saving:
- detailed_execution_data_TIMESTAMP.json (complete audit trail)
- baseline_results_detailed_TIMESTAMP.json (enhanced with dimensions/metadata)
- baseline_results_summary_TIMESTAMP.json (model performance summaries)

All data structures are properly serialized for JSON output with complete audit trails.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .metrics import (
    InputConsistencyData,
    ModelConsistencyData,
    EnhancedEvaluationResult,
)


class DataPersistenceManager:
    """Manages data persistence for Phase 1.5 enhanced evaluation results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the data persistence manager.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate consistent timestamp for all files in this session
        self.timestamp = datetime.now()

    def get_timestamp_string(self) -> str:
        """Get consistent timestamp string for file naming."""
        return self.timestamp.strftime("%Y%m%d_%H%M%S")

    def save_detailed_execution_data(
        self,
        experiment_metadata: Dict[str, Any],
        model_consistency_data: Dict[str, ModelConsistencyData],
    ) -> Path:
        """
        Save detailed_execution_data_TIMESTAMP.json with complete audit trail.

        Args:
            experiment_metadata: Experiment configuration and metadata
            model_consistency_data: Dict mapping model names to ModelConsistencyData

        Returns:
            Path to the saved file
        """
        # Build the complete data structure
        output_data = {"experiment_metadata": experiment_metadata}

        # Add per-model analysis
        for model_name, consistency_data in model_consistency_data.items():
            model_data = {
                "per_input_analysis": [
                    input_data.to_serializable_dict()
                    for input_data in consistency_data.input_consistency_data
                ],
                "model_summary": {
                    "overall_consistency": consistency_data.overall_consistency,
                    "dimensional_consistency": consistency_data.dimensional_consistency,
                    "per_input_consistency": consistency_data.per_input_consistency,
                    "total_inputs": consistency_data.total_inputs,
                    "total_variants": consistency_data.total_variants,
                },
            }
            output_data[model_name] = model_data

        # Save to file
        filename = f"detailed_execution_data_{self.get_timestamp_string()}.json"
        file_path = self.output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return file_path

    def save_enhanced_detailed_results(
        self, enhanced_results: Dict[str, Dict[str, EnhancedEvaluationResult]]
    ) -> Path:
        """
        Save baseline_results_detailed_TIMESTAMP.json with enhanced evaluation data.

        Args:
            enhanced_results: Dict mapping model names to variant results

        Returns:
            Path to the saved file
        """
        output_data = {}

        for model_name, model_results in enhanced_results.items():
            model_data = {}

            for variant_id, result in model_results.items():
                model_data[variant_id] = result.to_serializable_dict()

            output_data[model_name] = model_data

        # Save to file
        filename = f"baseline_results_detailed_{self.get_timestamp_string()}.json"
        file_path = self.output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return file_path

    def save_enhanced_summary_results(
        self, summary_results: Dict[str, Dict[str, Any]]
    ) -> Path:
        """
        Save baseline_results_summary_TIMESTAMP.json with performance summaries.

        Args:
            summary_results: Dict mapping model names to summary data

        Returns:
            Path to the saved file
        """
        # Save to file
        filename = f"baseline_results_summary_{self.get_timestamp_string()}.json"
        file_path = self.output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)

        return file_path

    def calculate_performance_summary(
        self, enhanced_results: Dict[str, Dict[str, EnhancedEvaluationResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance summaries from enhanced results.

        Args:
            enhanced_results: Dict mapping model names to variant results

        Returns:
            Summary data for each model
        """
        summary_results = {}

        for model_name, model_results in enhanced_results.items():
            if not model_results:
                continue

            # Extract metrics from all variants
            accuracies = [result.custom_accuracy for result in model_results.values()]
            consistencies = [
                result.consistency_score for result in model_results.values()
            ]
            weighted_indices = [
                result.weighted_index for result in model_results.values()
            ]

            # Calculate statistics
            accuracy_stats = {
                "mean": statistics.mean(accuracies),
                "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min": min(accuracies),
                "max": max(accuracies),
            }

            consistency_stats = {
                "mean": statistics.mean(consistencies),
                "std": statistics.stdev(consistencies)
                if len(consistencies) > 1
                else 0.0,
                "min": min(consistencies),
                "max": max(consistencies),
            }

            weighted_index_stats = {
                "mean": statistics.mean(weighted_indices),
                "std": statistics.stdev(weighted_indices)
                if len(weighted_indices) > 1
                else 0.0,
                "min": min(weighted_indices),
                "max": max(weighted_indices),
            }

            # Find best combination
            best_variant_id = max(
                model_results.keys(), key=lambda v: model_results[v].weighted_index
            )
            best_result = model_results[best_variant_id]

            # Calculate dimensional analysis
            dimensional_analysis = self._calculate_dimensional_analysis(model_results)

            summary_results[model_name] = {
                "performance_summary": {
                    "total_combinations": len(model_results),
                    "accuracy_stats": accuracy_stats,
                    "consistency_stats": consistency_stats,
                    "weighted_index_stats": weighted_index_stats,
                },
                "best_combination": {
                    "variant_id": best_variant_id,
                    "custom_accuracy": best_result.custom_accuracy,
                    "consistency_score": best_result.consistency_score,
                    "weighted_index": best_result.weighted_index,
                },
                "dimensional_analysis": dimensional_analysis,
                "total_variants_tested": len(model_results),
            }

        return summary_results

    def _calculate_dimensional_analysis(
        self, model_results: Dict[str, EnhancedEvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance analysis by dimension."""
        dimensional_analysis = {}

        # Group results by each dimension
        for dimension in ["formality", "phrasing", "order", "synonyms"]:
            dimension_groups = {}

            for variant_id, result in model_results.items():
                if result.variant_dimensions and dimension in result.variant_dimensions:
                    dim_value = result.variant_dimensions[dimension]

                    if dim_value not in dimension_groups:
                        dimension_groups[dim_value] = []

                    dimension_groups[dim_value].append(result.custom_accuracy)

            # Calculate average performance for each dimension value
            dimension_performance = {}
            for dim_value, accuracies in dimension_groups.items():
                if accuracies:
                    dimension_performance[dim_value] = statistics.mean(accuracies)

            if dimension_performance:
                dimensional_analysis[dimension] = dimension_performance

        return dimensional_analysis

    def save_complete_results(
        self,
        experiment_metadata: Dict[str, Any],
        model_consistency_data: Dict[str, ModelConsistencyData],
        enhanced_results: Dict[str, Dict[str, EnhancedEvaluationResult]],
    ) -> Dict[str, Path]:
        """
        Save all three result files with consistent timestamps.

        Args:
            experiment_metadata: Experiment configuration and metadata
            model_consistency_data: Model-level consistency data
            enhanced_results: Enhanced evaluation results

        Returns:
            Dict mapping file types to their paths
        """
        # Calculate summary results
        summary_results = self.calculate_performance_summary(enhanced_results)

        # Save all files
        file_paths = {
            "detailed_execution_data": self.save_detailed_execution_data(
                experiment_metadata, model_consistency_data
            ),
            "detailed_results": self.save_enhanced_detailed_results(enhanced_results),
            "summary_results": self.save_enhanced_summary_results(summary_results),
        }

        return file_paths
