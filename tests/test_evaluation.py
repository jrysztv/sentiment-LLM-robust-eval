"""
Tests for evaluation metrics and robustness analysis.
"""

import pytest
from unittest.mock import Mock
import json

from deep_learning_final_assignment.core.evaluation.metrics import (
    CustomAccuracyMetric,
    ConsistencyMetric,
    RobustnessEvaluator,
    EvaluationResult,
)
from deep_learning_final_assignment.core.models.base import ModelResponse
from deep_learning_final_assignment.core.data.loaders import DataSample


class TestCustomAccuracyMetric:
    """Test the CustomAccuracyMetric class."""

    def test_init(self):
        """Test CustomAccuracyMetric initialization."""
        metric = CustomAccuracyMetric()

        assert metric.label_to_encoding is not None
        assert metric.encoding_to_label is not None
        assert len(metric.label_to_encoding) == 5

    def test_calculate_penalty_correct(self):
        """Test penalty calculation for correct predictions."""
        metric = CustomAccuracyMetric()

        penalty = metric.calculate_penalty("Positive", "Positive")
        assert penalty == 0.0

    def test_calculate_penalty_adjacent(self):
        """Test penalty calculation for adjacent errors."""
        metric = CustomAccuracyMetric()

        # Adjacent errors should have penalty of 1
        penalty1 = metric.calculate_penalty("Negative", "Very Negative")
        penalty2 = metric.calculate_penalty("Very Positive", "Positive")

        assert penalty1 == 1.0
        assert penalty2 == 1.0

    def test_calculate_penalty_base_neutral(self):
        """Test penalty calculation for base-neutral errors."""
        metric = CustomAccuracyMetric()

        # Base to neutral errors should have penalty of 4
        penalty1 = metric.calculate_penalty("Negative", "Neutral")
        penalty2 = metric.calculate_penalty("Positive", "Neutral")

        assert penalty1 == 4.0
        assert penalty2 == 4.0

    def test_calculate_penalty_cross_polarity(self):
        """Test penalty calculation for cross-polarity errors."""
        metric = CustomAccuracyMetric()

        # Cross-polarity errors should have penalty of 16
        penalty1 = metric.calculate_penalty("Negative", "Positive")
        penalty2 = metric.calculate_penalty("Positive", "Negative")

        assert penalty1 == 16.0
        assert penalty2 == 16.0

    def test_calculate_penalty_extreme(self):
        """Test penalty calculation for extreme errors."""
        metric = CustomAccuracyMetric()

        # Extreme errors should have penalty of 36
        penalty1 = metric.calculate_penalty("Very Negative", "Very Positive")
        penalty2 = metric.calculate_penalty("Very Positive", "Very Negative")

        assert penalty1 == 36.0
        assert penalty2 == 36.0

    def test_calculate_penalty_invalid(self):
        """Test penalty calculation for invalid predictions."""
        metric = CustomAccuracyMetric()

        penalty = metric.calculate_penalty("Invalid Label", "Positive")
        assert penalty == 36.0  # Maximum penalty

    def test_calculate_accuracy_perfect(self):
        """Test accuracy calculation for perfect predictions."""
        metric = CustomAccuracyMetric()

        predictions = [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ]
        true_labels = [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ]

        accuracy = metric.calculate_accuracy(predictions, true_labels)
        assert accuracy == 1.0

    def test_calculate_accuracy_worst(self):
        """Test accuracy calculation for worst predictions."""
        metric = CustomAccuracyMetric()

        predictions = [
            "Very Positive",
            "Very Positive",
            "Very Positive",
            "Very Negative",
            "Very Negative",
        ]
        true_labels = [
            "Very Negative",
            "Very Negative",
            "Very Negative",
            "Very Positive",
            "Very Positive",
        ]

        accuracy = metric.calculate_accuracy(predictions, true_labels)
        assert accuracy == 0.0

    def test_calculate_accuracy_mixed(self):
        """Test accuracy calculation for mixed predictions."""
        metric = CustomAccuracyMetric()

        predictions = ["Positive", "Negative", "Neutral"]  # 0, 4, 0 penalties
        true_labels = ["Positive", "Neutral", "Neutral"]

        total_penalty = 0 + 4 + 0  # = 4
        max_penalty = 3 * 36  # = 108
        expected_accuracy = 1.0 - (4 / 108)

        accuracy = metric.calculate_accuracy(predictions, true_labels)
        assert abs(accuracy - expected_accuracy) < 1e-6

    def test_calculate_accuracy_empty(self):
        """Test accuracy calculation for empty lists."""
        metric = CustomAccuracyMetric()

        accuracy = metric.calculate_accuracy([], [])
        assert accuracy == 0.0

    def test_calculate_accuracy_mismatched_length(self):
        """Test error handling for mismatched list lengths."""
        metric = CustomAccuracyMetric()

        with pytest.raises(ValueError, match="must have the same length"):
            metric.calculate_accuracy(["Positive"], ["Positive", "Negative"])

    def test_get_error_breakdown(self):
        """Test error breakdown calculation."""
        metric = CustomAccuracyMetric()

        predictions = [
            "Positive",  # Correct (0 penalty)
            "Very Negative",  # Adjacent to Negative (1 penalty)
            "Neutral",  # Base to neutral from Positive (4 penalty)
            "Positive",  # Cross-polarity from Negative (16 penalty)
            "Very Positive",  # Extreme from Very Negative (36 penalty)
            "Invalid",  # Invalid prediction
        ]
        true_labels = [
            "Positive",
            "Negative",
            "Positive",
            "Negative",
            "Very Negative",
            "Neutral",
        ]

        breakdown = metric.get_error_breakdown(predictions, true_labels)

        assert breakdown["correct_predictions"] == 1
        assert breakdown["adjacent_errors"] == 1
        assert breakdown["base_neutral_errors"] == 1
        assert breakdown["cross_polarity_errors"] == 1
        assert breakdown["extreme_errors"] == 1
        assert breakdown["invalid_predictions"] == 1


class TestConsistencyMetric:
    """Test the ConsistencyMetric class."""

    def test_calculate_consistency_perfect(self):
        """Test consistency calculation for perfect consistency."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Negative", "Neutral"],
            "v2": ["Positive", "Negative", "Neutral"],
            "v3": ["Positive", "Negative", "Neutral"],
        }

        consistency = metric.calculate_consistency(variant_predictions)
        assert consistency == 1.0

    def test_calculate_consistency_no_consistency(self):
        """Test consistency calculation for no consistency."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Positive", "Positive"],
            "v2": ["Negative", "Negative", "Negative"],
            "v3": ["Neutral", "Neutral", "Neutral"],
        }

        consistency = metric.calculate_consistency(variant_predictions)
        assert consistency == 0.0

    def test_calculate_consistency_partial(self):
        """Test consistency calculation for partial consistency."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Negative", "Neutral"],
            "v2": ["Positive", "Positive", "Neutral"],  # 2/3 consistent
        }

        consistency = metric.calculate_consistency(variant_predictions)
        assert abs(consistency - (2 / 3)) < 1e-6

    def test_calculate_consistency_empty(self):
        """Test consistency calculation for empty predictions."""
        metric = ConsistencyMetric()

        consistency = metric.calculate_consistency({})
        assert consistency == 0.0

    def test_calculate_consistency_mismatched_length(self):
        """Test error handling for mismatched prediction lengths."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Negative"],
            "v2": ["Positive"],  # Different length
        }

        with pytest.raises(ValueError, match="must have the same number"):
            metric.calculate_consistency(variant_predictions)

    def test_calculate_pairwise_consistency(self):
        """Test pairwise consistency calculation."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Negative", "Neutral"],
            "v2": ["Positive", "Positive", "Neutral"],  # 2/3 with v1
            "v3": ["Positive", "Negative", "Positive"],  # 2/3 with v1, 1/3 with v2
        }

        pairwise = metric.calculate_pairwise_consistency(variant_predictions)

        assert abs(pairwise[("v1", "v2")] - (2 / 3)) < 1e-6
        assert abs(pairwise[("v1", "v3")] - (2 / 3)) < 1e-6
        assert abs(pairwise[("v2", "v3")] - (1 / 3)) < 1e-6

    def test_calculate_dimensional_consistency(self):
        """Test dimensional consistency calculation."""
        metric = ConsistencyMetric()

        variant_predictions = {
            "v1": ["Positive", "Negative"],
            "v2": ["Positive", "Negative"],  # Same formality as v1
            "v3": ["Positive", "Positive"],  # Different formality
            "v4": ["Negative", "Positive"],  # Different formality
        }

        variant_dimensions = {
            "v1": {"formality": "formal", "phrasing": "imperative"},
            "v2": {"formality": "formal", "phrasing": "question"},
            "v3": {"formality": "casual", "phrasing": "imperative"},
            "v4": {"formality": "casual", "phrasing": "question"},
        }

        dimensional = metric.calculate_dimensional_consistency(
            variant_predictions, variant_dimensions
        )

        assert "formality" in dimensional
        assert "phrasing" in dimensional
        # Formal variants (v1, v2) have 100% consistency
        # Casual variants (v3, v4) have 0% consistency
        # So formality dimension should have (1.0 + 0.0) / 2 = 0.5


class TestRobustnessEvaluator:
    """Test the RobustnessEvaluator class."""

    def test_init(self):
        """Test RobustnessEvaluator initialization."""
        evaluator = RobustnessEvaluator()

        assert evaluator.accuracy_metric is not None
        assert evaluator.consistency_metric is not None

    def test_parse_model_response_valid_json(self):
        """Test parsing valid JSON response."""
        evaluator = RobustnessEvaluator()

        response = ModelResponse(
            content='{"sentiment": "Positive"}',
            model_name="test_model",
            prompt_used="test_prompt",
        )

        parsed = evaluator.parse_model_response(response)
        assert parsed == "Positive"

    def test_parse_model_response_malformed_json(self):
        """Test parsing malformed JSON response."""
        evaluator = RobustnessEvaluator()

        response = ModelResponse(
            content="The sentiment is Positive",
            model_name="test_model",
            prompt_used="test_prompt",
        )

        parsed = evaluator.parse_model_response(response)
        assert parsed == "Positive"

    def test_parse_model_response_embedded_json(self):
        """Test parsing response with embedded JSON."""
        evaluator = RobustnessEvaluator()

        response = ModelResponse(
            content='Here is my analysis: {"sentiment": "Negative"} based on the text.',
            model_name="test_model",
            prompt_used="test_prompt",
        )

        parsed = evaluator.parse_model_response(response)
        assert parsed == "Negative"

    def test_parse_model_response_no_sentiment(self):
        """Test parsing response with no recognizable sentiment."""
        evaluator = RobustnessEvaluator()

        response = ModelResponse(
            content="I cannot determine the sentiment.",
            model_name="test_model",
            prompt_used="test_prompt",
        )

        parsed = evaluator.parse_model_response(response)
        assert parsed == "Unknown"

    def test_evaluate_single_combination(
        self, sample_model_responses, sample_data_samples
    ):
        """Test evaluating a single model-prompt combination."""
        evaluator = RobustnessEvaluator()

        result = evaluator.evaluate_single_combination(
            sample_model_responses, sample_data_samples, "v1"
        )

        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"
        assert result.prompt_variant_id == "v1"
        assert 0.0 <= result.custom_accuracy <= 1.0
        assert result.consistency_score == 1.0  # Single combination
        assert 0.0 <= result.weighted_index <= 1.0
        assert len(result.predictions) == 5
        assert len(result.true_labels) == 5

    def test_evaluate_single_combination_mismatched_length(
        self, sample_model_responses, sample_data_samples
    ):
        """Test error handling for mismatched response and sample lengths."""
        evaluator = RobustnessEvaluator()

        with pytest.raises(ValueError, match="must match number of true samples"):
            evaluator.evaluate_single_combination(
                sample_model_responses[:3], sample_data_samples, "v1"
            )

    def test_evaluate_multiple_combinations(self, sample_data_samples):
        """Test evaluating multiple model-prompt combinations."""
        evaluator = RobustnessEvaluator()

        # Create mock responses for multiple variants
        combination_responses = {}
        for variant_id in ["v1", "v2", "v3"]:
            responses = []
            for sample in sample_data_samples:
                response = ModelResponse(
                    content=f'{{"sentiment": "{sample.label}"}}',  # Perfect predictions
                    model_name="test_model",
                    prompt_used=f"prompt_{variant_id}",
                )
                responses.append(response)
            combination_responses[variant_id] = responses

        variant_dimensions = {
            "v1": {"formality": "formal", "phrasing": "imperative"},
            "v2": {"formality": "formal", "phrasing": "question"},
            "v3": {"formality": "casual", "phrasing": "imperative"},
        }

        results = evaluator.evaluate_multiple_combinations(
            combination_responses, sample_data_samples, variant_dimensions
        )

        assert len(results) == 3
        assert all(isinstance(result, EvaluationResult) for result in results.values())

        # All should have perfect accuracy since we used correct labels
        for result in results.values():
            assert result.custom_accuracy == 1.0
            assert result.consistency_score == 1.0  # Perfect consistency too

    def test_get_best_combination(self):
        """Test getting the best performing combination."""
        evaluator = RobustnessEvaluator()

        results = {
            "v1": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v1",
                custom_accuracy=0.8,
                consistency_score=0.9,
                weighted_index=0.83,  # 0.7*0.8 + 0.3*0.9
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
            "v2": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v2",
                custom_accuracy=0.9,
                consistency_score=0.7,
                weighted_index=0.84,  # 0.7*0.9 + 0.3*0.7
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
        }

        best_id, best_result = evaluator.get_best_combination(results)

        assert best_id == "v2"
        assert best_result.weighted_index == 0.84

    def test_get_best_combination_empty(self):
        """Test error handling for empty results."""
        evaluator = RobustnessEvaluator()

        with pytest.raises(ValueError, match="No results to evaluate"):
            evaluator.get_best_combination({})

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        evaluator = RobustnessEvaluator()

        results = {
            "v1": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v1",
                custom_accuracy=0.8,
                consistency_score=0.9,
                weighted_index=0.83,
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
            "v2": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v2",
                custom_accuracy=0.9,
                consistency_score=0.7,
                weighted_index=0.84,
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
        }

        summary = evaluator.get_performance_summary(results)

        assert summary["total_combinations"] == 2
        assert "accuracy_stats" in summary
        assert "consistency_stats" in summary
        assert "weighted_index_stats" in summary

        assert abs(summary["accuracy_stats"]["mean"] - 0.85) < 1e-10  # (0.8 + 0.9) / 2
        assert summary["accuracy_stats"]["min"] == 0.8
        assert summary["accuracy_stats"]["max"] == 0.9

    def test_get_performance_summary_empty(self):
        """Test performance summary for empty results."""
        evaluator = RobustnessEvaluator()

        summary = evaluator.get_performance_summary({})
        assert summary == {}


class TestEvaluationResult:
    """Test the EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            model_name="test_model",
            prompt_variant_id="v1",
            custom_accuracy=0.85,
            consistency_score=0.92,
            weighted_index=0.871,
            predictions=["Positive", "Negative"],
            true_labels=["Positive", "Neutral"],
            error_breakdown={"correct": 1, "errors": 1},
            metadata={"test": "value"},
        )

        assert result.model_name == "test_model"
        assert result.prompt_variant_id == "v1"
        assert result.custom_accuracy == 0.85
        assert result.consistency_score == 0.92
        assert result.weighted_index == 0.871
        assert result.predictions == ["Positive", "Negative"]
        assert result.true_labels == ["Positive", "Neutral"]
        assert result.error_breakdown == {"correct": 1, "errors": 1}
        assert result.metadata == {"test": "value"}

    def test_evaluation_result_default_metadata(self):
        """Test EvaluationResult with default metadata."""
        result = EvaluationResult(
            model_name="test_model",
            prompt_variant_id="v1",
            custom_accuracy=0.85,
            consistency_score=0.92,
            weighted_index=0.871,
            predictions=[],
            true_labels=[],
            error_breakdown={},
        )

        assert result.metadata == {}
