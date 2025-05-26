"""
Custom evaluation metrics for prompt robustness testing.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
from sklearn.metrics import mean_squared_error

from ..data.loaders import SentimentLabels, DataSample
from ..models.base import ModelResponse


@dataclass
class EvaluationResult:
    """Results from evaluating a model-prompt combination."""

    model_name: str
    prompt_variant_id: str
    custom_accuracy: float
    consistency_score: float
    weighted_index: float
    predictions: List[str]
    true_labels: List[str]
    error_breakdown: Dict[str, int]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InputConsistencyData:
    """Complete analysis data for a single input across all prompt variants."""

    input_id: int
    input_text: str
    true_label: str
    variant_predictions: Dict[str, str]  # variant_id -> prediction
    variant_dimensions: Dict[str, Dict[str, str]]  # variant_id -> dimensions
    prediction_distribution: Dict[str, float]  # label -> percentage
    consistency_score: float  # max(prediction_distribution.values())
    majority_label: str  # most frequent prediction
    formatted_prompts: Dict[str, str]  # variant_id -> formatted_prompt
    raw_responses: Dict[str, ModelResponse]  # variant_id -> raw_response

    @staticmethod
    def calculate_prediction_distribution(
        variant_predictions: Dict[str, str],
    ) -> Dict[str, float]:
        """Calculate prediction distribution from variant predictions."""
        if not variant_predictions:
            return {}

        # Count predictions
        prediction_counts = Counter(variant_predictions.values())
        total_variants = len(variant_predictions)

        # Create distribution for all possible labels
        distribution = {}
        for label in SentimentLabels.get_all_labels():
            distribution[label] = prediction_counts.get(label, 0) / total_variants

        return distribution

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for JSON output."""
        data_dict = asdict(self)

        # Convert ModelResponse objects to serializable format
        data_dict["raw_responses"] = {
            variant_id: {
                "content": response.content,
                "model_name": response.model_name,
                "prompt_used": response.prompt_used,
                "metadata": getattr(response, "metadata", {}),
            }
            for variant_id, response in self.raw_responses.items()
        }

        return data_dict


@dataclass
class ModelConsistencyData:
    """Model-level consistency analysis aggregated from all inputs."""

    model_name: str
    overall_consistency: float  # mean(per_input_consistency)
    per_input_consistency: List[float]  # consistency score per input
    input_consistency_data: List[InputConsistencyData]
    dimensional_consistency: Dict[str, float]  # consistency by dimension
    total_inputs: int
    total_variants: int

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for JSON output."""
        data_dict = asdict(self)

        # Convert InputConsistencyData objects to serializable format
        data_dict["input_consistency_data"] = [
            input_data.to_serializable_dict()
            for input_data in self.input_consistency_data
        ]

        return data_dict


@dataclass
class EnhancedEvaluationResult(EvaluationResult):
    """Extended evaluation result with complete data persistence."""

    variant_dimensions: Dict[str, str] = None
    variant_name: str = None
    formatted_prompts: List[str] = None
    raw_responses: List[ModelResponse] = None
    execution_metadata: Dict[str, Any] = None

    def __post_init__(self):
        super().__post_init__()
        if self.variant_dimensions is None:
            self.variant_dimensions = {}
        if self.formatted_prompts is None:
            self.formatted_prompts = []
        if self.raw_responses is None:
            self.raw_responses = []
        if self.execution_metadata is None:
            self.execution_metadata = {}

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for JSON output."""
        data_dict = asdict(self)

        # Convert ModelResponse objects to serializable format
        data_dict["raw_responses"] = [
            {
                "content": response.content,
                "model_name": response.model_name,
                "prompt_used": response.prompt_used,
                "metadata": getattr(response, "metadata", {}),
            }
            for response in self.raw_responses
        ]

        return data_dict


class CustomAccuracyMetric:
    """
    Polarity-weighted encoding accuracy using sklearn's MSE implementation.

    Label Encoding: Very Negative=-3, Negative=-2, Neutral=0, Positive=2, Very Positive=3
    Penalty Calculation: sklearn.metrics.mean_squared_error(actual_encodings, predicted_encodings)

    Resulting Penalty Structure:
    - Adjacent Errors (1 point): Negative ↔ Very Negative, Positive ↔ Very Positive
    - Base→Neutral Errors (4 points): Negative ↔ Neutral, Positive ↔ Neutral
    - Cross-Polarity Errors (16 points): Negative ↔ Positive
    - Extreme Errors (36 points): Very Negative ↔ Very Positive
    """

    def __init__(self):
        self.label_to_encoding = SentimentLabels.get_label_to_encoding_map()
        self.encoding_to_label = SentimentLabels.get_encoding_to_label_map()
        self.max_penalty_per_sample = (
            36.0  # Maximum possible MSE (Very Negative ↔ Very Positive)
        )

    def _encode_labels(self, labels: List[str]) -> List[float]:
        """Convert sentiment labels to numerical encodings."""
        encodings = []
        for label in labels:
            if label in self.label_to_encoding:
                encodings.append(float(self.label_to_encoding[label]))
            else:
                # For invalid labels, use encoding that gives maximum penalty
                # We'll use the most extreme encoding to ensure maximum MSE
                encodings.append(float(self.label_to_encoding["Very Positive"]))
        return encodings

    def calculate_penalty(self, predicted: str, actual: str) -> float:
        """Calculate MSE penalty for a single prediction using sklearn."""
        try:
            pred_encoding = float(self.label_to_encoding[predicted])
            actual_encoding = float(self.label_to_encoding[actual])
            # Use sklearn's MSE for single sample (returns the squared difference)
            return mean_squared_error([actual_encoding], [pred_encoding])
        except KeyError:
            # If prediction is not a valid label, assign maximum penalty
            return self.max_penalty_per_sample

    def calculate_accuracy(
        self, predictions: List[str], true_labels: List[str]
    ) -> float:
        """
        Calculate custom accuracy score using sklearn's MSE.

        Args:
            predictions: List of predicted sentiment labels
            true_labels: List of true sentiment labels

        Returns:
            Accuracy score between 0 and 1 (higher is better)
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true labels must have the same length")

        if not predictions:
            return 0.0

        # Convert labels to numerical encodings
        true_encodings = self._encode_labels(true_labels)
        pred_encodings = self._encode_labels(predictions)

        # Calculate MSE using sklearn
        mse_penalty = mean_squared_error(true_encodings, pred_encodings)

        # Convert MSE to accuracy (0-1 scale)
        # Maximum possible MSE per sample is 36, so normalize by that
        accuracy = 1.0 - (mse_penalty / self.max_penalty_per_sample)

        return max(0.0, accuracy)  # Ensure non-negative

    def get_error_breakdown(
        self, predictions: List[str], true_labels: List[str]
    ) -> Dict[str, int]:
        """Get breakdown of error types."""
        error_counts = {
            "adjacent_errors": 0,  # 1 point penalties
            "base_neutral_errors": 0,  # 4 point penalties
            "cross_polarity_errors": 0,  # 16 point penalties
            "extreme_errors": 0,  # 36 point penalties
            "invalid_predictions": 0,  # Invalid label predictions
            "correct_predictions": 0,  # 0 point penalties
        }

        for pred, true in zip(predictions, true_labels):
            penalty = self.calculate_penalty(pred, true)

            if pred not in self.label_to_encoding:
                error_counts["invalid_predictions"] += 1
            elif penalty == 0:
                error_counts["correct_predictions"] += 1
            elif penalty == 1:
                error_counts["adjacent_errors"] += 1
            elif penalty == 4:
                error_counts["base_neutral_errors"] += 1
            elif penalty == 16:
                error_counts["cross_polarity_errors"] += 1
            elif penalty == 36:
                error_counts["extreme_errors"] += 1

        return error_counts


class ConsistencyMetric:
    """Measures consistency of predictions across prompt variants."""

    def calculate_consistency(self, variant_predictions: Dict[str, List[str]]) -> float:
        """
        Calculate consistency score across prompt variants.

        Args:
            variant_predictions: Dict mapping variant IDs to lists of predictions

        Returns:
            Consistency score between 0 and 1 (higher is better)
        """
        if not variant_predictions:
            return 0.0

        # Get all variant prediction lists
        prediction_lists = list(variant_predictions.values())

        if not prediction_lists or not prediction_lists[0]:
            return 0.0

        n_samples = len(prediction_lists[0])

        # Check that all variants have the same number of predictions
        if not all(len(preds) == n_samples for preds in prediction_lists):
            raise ValueError("All variants must have the same number of predictions")

        consistent_predictions = 0

        for i in range(n_samples):
            # Get predictions for sample i across all variants
            sample_predictions = [preds[i] for preds in prediction_lists]

            # Check if all predictions are the same
            if len(set(sample_predictions)) == 1:
                consistent_predictions += 1

        return consistent_predictions / n_samples

    def calculate_pairwise_consistency(
        self, variant_predictions: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise consistency between all variant pairs."""
        variants = list(variant_predictions.keys())
        pairwise_scores = {}

        for i, variant1 in enumerate(variants):
            for variant2 in variants[i + 1 :]:
                preds1 = variant_predictions[variant1]
                preds2 = variant_predictions[variant2]

                if len(preds1) != len(preds2):
                    continue

                consistent = sum(1 for p1, p2 in zip(preds1, preds2) if p1 == p2)
                score = consistent / len(preds1) if preds1 else 0.0
                pairwise_scores[(variant1, variant2)] = score

        return pairwise_scores

    def calculate_dimensional_consistency(
        self,
        variant_predictions: Dict[str, List[str]],
        variant_dimensions: Dict[str, Dict[str, str]],
    ) -> Dict[str, float]:
        """Calculate consistency within each dimension."""
        dimension_scores = {}

        # Group variants by each dimension
        for dimension in ["formality", "phrasing", "order", "synonyms"]:
            dimension_groups = defaultdict(list)

            for variant_id, predictions in variant_predictions.items():
                if variant_id in variant_dimensions:
                    dim_value = variant_dimensions[variant_id].get(dimension)
                    if dim_value:
                        dimension_groups[dim_value].append(predictions)

            # Calculate consistency within each dimension value
            dimension_consistency_scores = []
            for dim_value, prediction_lists in dimension_groups.items():
                if len(prediction_lists) > 1:
                    # Create a temporary dict for this dimension group
                    temp_dict = {
                        f"{dimension}_{dim_value}_{i}": preds
                        for i, preds in enumerate(prediction_lists)
                    }
                    consistency = self.calculate_consistency(temp_dict)
                    dimension_consistency_scores.append(consistency)

            # Average consistency across dimension values
            if dimension_consistency_scores:
                dimension_scores[dimension] = statistics.mean(
                    dimension_consistency_scores
                )
            else:
                dimension_scores[dimension] = 0.0

        return dimension_scores


class EnhancedConsistencyMetric:
    """Enhanced consistency metric with corrected calculation logic."""

    def calculate_per_input_consistency(
        self, variant_predictions_for_input: List[str]
    ) -> float:
        """
        Calculate per-input consistency as max(prediction_distribution).

        Args:
            variant_predictions_for_input: List of predictions from all variants for one input

        Returns:
            Consistency score between 0 and 1 (percentage of variants that agree on majority prediction)
        """
        if not variant_predictions_for_input:
            return 0.0

        # Count predictions
        prediction_counts = Counter(variant_predictions_for_input)
        total_variants = len(variant_predictions_for_input)

        # Return the percentage of variants that agree on the most common prediction
        max_count = max(prediction_counts.values())
        return max_count / total_variants

    def calculate_model_consistency(
        self, per_input_consistency_scores: List[float]
    ) -> float:
        """
        Calculate model consistency as mean of per-input consistency scores.

        Args:
            per_input_consistency_scores: List of consistency scores per input

        Returns:
            Overall model consistency score
        """
        if not per_input_consistency_scores:
            return 0.0

        return statistics.mean(per_input_consistency_scores)

    def calculate_enhanced_consistency(
        self, variant_predictions: Dict[str, List[str]]
    ) -> ModelConsistencyData:
        """
        Calculate complete consistency analysis using corrected logic.

        Args:
            variant_predictions: Dict mapping variant IDs to lists of predictions

        Returns:
            ModelConsistencyData with complete analysis
        """
        if not variant_predictions:
            return ModelConsistencyData(
                model_name="Unknown",
                overall_consistency=0.0,
                per_input_consistency=[],
                input_consistency_data=[],
                dimensional_consistency={},
                total_inputs=0,
                total_variants=0,
            )

        # Get number of inputs and variants
        variant_lists = list(variant_predictions.values())
        total_inputs = len(variant_lists[0]) if variant_lists else 0
        total_variants = len(variant_predictions)

        # Validate all variants have same number of predictions
        if not all(len(preds) == total_inputs for preds in variant_lists):
            raise ValueError("All variants must have the same number of predictions")

        # Calculate per-input consistency
        per_input_consistency = []

        for input_idx in range(total_inputs):
            # Get predictions for this input across all variants
            input_predictions = [
                variant_predictions[variant_id][input_idx]
                for variant_id in variant_predictions
            ]
            input_consistency = self.calculate_per_input_consistency(input_predictions)
            per_input_consistency.append(input_consistency)

        # Calculate overall model consistency
        overall_consistency = self.calculate_model_consistency(per_input_consistency)

        return ModelConsistencyData(
            model_name="Unknown",  # Will be set by caller
            overall_consistency=overall_consistency,
            per_input_consistency=per_input_consistency,
            input_consistency_data=[],  # Will be populated by caller with full data
            dimensional_consistency={},  # Will be calculated by caller
            total_inputs=total_inputs,
            total_variants=total_variants,
        )

    def calculate_dimensional_consistency(
        self,
        variant_predictions: Dict[str, List[str]],
        variant_dimensions: Dict[str, Dict[str, str]],
    ) -> Dict[str, float]:
        """Calculate consistency within each dimension."""
        dimension_scores = {}

        # Group variants by each dimension
        for dimension in ["formality", "phrasing", "order", "synonyms"]:
            dimension_groups = {}

            for variant_id, predictions in variant_predictions.items():
                if variant_id in variant_dimensions:
                    dim_value = variant_dimensions[variant_id].get(dimension)
                    if dim_value:
                        if dim_value not in dimension_groups:
                            dimension_groups[dim_value] = []
                        dimension_groups[dim_value].append(predictions)

            # Calculate consistency within each dimension value
            dimension_consistency_scores = []
            for dim_value, prediction_lists in dimension_groups.items():
                if len(prediction_lists) > 1:
                    # Create temporary dict for this dimension group
                    temp_variant_predictions = {
                        f"{dimension}_{dim_value}_{i}": preds
                        for i, preds in enumerate(prediction_lists)
                    }
                    consistency_data = self.calculate_enhanced_consistency(
                        temp_variant_predictions
                    )
                    dimension_consistency_scores.append(
                        consistency_data.overall_consistency
                    )

            # Average consistency across dimension values
            if dimension_consistency_scores:
                dimension_scores[dimension] = statistics.mean(
                    dimension_consistency_scores
                )
            else:
                dimension_scores[dimension] = 0.0

        return dimension_scores


class RobustnessEvaluator:
    """Comprehensive robustness evaluation system."""

    def __init__(self):
        self.accuracy_metric = CustomAccuracyMetric()
        self.consistency_metric = ConsistencyMetric()

    def parse_model_response(self, response: ModelResponse) -> str:
        """
        Parse model response to extract sentiment prediction.

        Args:
            response: ModelResponse object

        Returns:
            Predicted sentiment label
        """
        try:
            # Try to parse as JSON first
            content = response.content.strip()

            # Handle common JSON formatting issues
            if not content.startswith("{"):
                # Look for JSON-like content in the response
                import re

                json_match = re.search(r'\{[^}]*"sentiment"[^}]*\}', content)
                if json_match:
                    content = json_match.group()
                else:
                    # Fallback: look for sentiment labels directly
                    for label in SentimentLabels.get_all_labels():
                        if label in content:
                            return label
                    return "Unknown"

            parsed = json.loads(content)

            # Extract sentiment from various possible keys
            for key in ["sentiment", "label", "prediction", "classification"]:
                if key in parsed:
                    return str(parsed[key])

            return "Unknown"

        except (json.JSONDecodeError, AttributeError):
            # Fallback: search for sentiment labels in the raw content
            content = response.content.lower()
            for label in SentimentLabels.get_all_labels():
                if label.lower() in content:
                    return label
            return "Unknown"

    def evaluate_single_combination(
        self,
        responses: List[ModelResponse],
        true_samples: List[DataSample],
        variant_id: str,
    ) -> EvaluationResult:
        """
        Evaluate a single model-prompt combination.

        Args:
            responses: List of model responses
            true_samples: List of true data samples
            variant_id: Prompt variant identifier

        Returns:
            EvaluationResult object
        """
        if len(responses) != len(true_samples):
            raise ValueError("Number of responses must match number of true samples")

        # Parse predictions
        predictions = [self.parse_model_response(response) for response in responses]
        true_labels = [sample.label for sample in true_samples]

        # Calculate metrics
        custom_accuracy = self.accuracy_metric.calculate_accuracy(
            predictions, true_labels
        )
        error_breakdown = self.accuracy_metric.get_error_breakdown(
            predictions, true_labels
        )

        # For single combination, consistency is 1.0 (consistent with itself)
        consistency_score = 1.0

        # Calculate weighted index (70% accuracy + 30% consistency)
        weighted_index = 0.7 * custom_accuracy + 0.3 * consistency_score

        # Extract model name from first response
        model_name = responses[0].model_name if responses else "Unknown"

        return EvaluationResult(
            model_name=model_name,
            prompt_variant_id=variant_id,
            custom_accuracy=custom_accuracy,
            consistency_score=consistency_score,
            weighted_index=weighted_index,
            predictions=predictions,
            true_labels=true_labels,
            error_breakdown=error_breakdown,
            metadata={
                "total_samples": len(responses),
                "response_metadata": [r.metadata for r in responses],
            },
        )

    def evaluate_multiple_combinations(
        self,
        combination_responses: Dict[str, List[ModelResponse]],
        true_samples: List[DataSample],
        variant_dimensions: Dict[str, Dict[str, str]],
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple model-prompt combinations.

        Args:
            combination_responses: Dict mapping variant IDs to lists of responses
            true_samples: List of true data samples
            variant_dimensions: Dict mapping variant IDs to their dimensions

        Returns:
            Dictionary mapping variant IDs to EvaluationResult objects
        """
        results = {}

        # First, evaluate each combination individually
        for variant_id, responses in combination_responses.items():
            results[variant_id] = self.evaluate_single_combination(
                responses, true_samples, variant_id
            )

        # Calculate cross-variant consistency
        variant_predictions = {
            variant_id: result.predictions for variant_id, result in results.items()
        }

        overall_consistency = self.consistency_metric.calculate_consistency(
            variant_predictions
        )
        dimensional_consistency = (
            self.consistency_metric.calculate_dimensional_consistency(
                variant_predictions, variant_dimensions
            )
        )

        # Update consistency scores and weighted indices
        for variant_id, result in results.items():
            result.consistency_score = overall_consistency
            result.weighted_index = (
                0.7 * result.custom_accuracy + 0.3 * overall_consistency
            )
            result.metadata["dimensional_consistency"] = dimensional_consistency
            result.metadata["overall_consistency"] = overall_consistency

        return results

    def get_best_combination(
        self, results: Dict[str, EvaluationResult]
    ) -> Tuple[str, EvaluationResult]:
        """Get the best performing model-prompt combination."""
        if not results:
            raise ValueError("No results to evaluate")

        best_variant_id = max(results.keys(), key=lambda k: results[k].weighted_index)
        return best_variant_id, results[best_variant_id]

    def get_best_combinations_per_model(
        self, all_results: Dict[str, Dict[str, EvaluationResult]]
    ) -> Dict[str, Tuple[str, EvaluationResult]]:
        """
        Get the best performing variant for each model separately.

        Args:
            all_results: Dictionary mapping model_name -> {variant_id -> EvaluationResult}

        Returns:
            Dictionary mapping model_name -> (best_variant_id, best_result)

        Raises:
            ValueError: If no results provided
        """
        if not all_results:
            raise ValueError("No results provided")

        best_per_model = {}
        for model_name, model_results in all_results.items():
            if model_results:  # Skip empty model results
                best_variant_id, best_result = self.get_best_combination(model_results)
                best_per_model[model_name] = (best_variant_id, best_result)

        return best_per_model

    def get_performance_summary(
        self, results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """Get summary statistics across all combinations."""
        if not results:
            return {}

        accuracies = [r.custom_accuracy for r in results.values()]
        consistencies = [r.consistency_score for r in results.values()]
        weighted_indices = [r.weighted_index for r in results.values()]

        return {
            "total_combinations": len(results),
            "accuracy_stats": {
                "mean": statistics.mean(accuracies),
                "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min": min(accuracies),
                "max": max(accuracies),
            },
            "consistency_stats": {
                "mean": statistics.mean(consistencies),
                "std": statistics.stdev(consistencies)
                if len(consistencies) > 1
                else 0.0,
                "min": min(consistencies),
                "max": max(consistencies),
            },
            "weighted_index_stats": {
                "mean": statistics.mean(weighted_indices),
                "std": statistics.stdev(weighted_indices)
                if len(weighted_indices) > 1
                else 0.0,
                "min": min(weighted_indices),
                "max": max(weighted_indices),
            },
        }
