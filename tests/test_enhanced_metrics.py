"""
Tests for Phase 1.5 enhanced data structures.

This module tests the new data structures that will capture:
- Test-set inputs and their labels
- Prompt variant dimensions from sentiment_prompts.py
- Complete LLM call outputs and metadata
- Evaluated metrics and consistency calculations
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from unittest.mock import Mock

from deep_learning_final_assignment.core.models.base import ModelResponse


class TestInputConsistencyData:
    """Test the InputConsistencyData structure for capturing per-input analysis."""

    def test_input_consistency_data_creation(self):
        """Test creating InputConsistencyData with complete data."""
        # This test will fail initially - we need to implement the dataclass
        from deep_learning_final_assignment.core.evaluation.metrics import (
            InputConsistencyData,
        )

        # Mock data representing what we'll capture
        variant_predictions = {
            "v1": "Positive",
            "v2": "Very Positive",
            "v3": "Positive",
        }

        variant_dimensions = {
            "v1": {
                "formality": "formal",
                "phrasing": "imperative",
                "order": "task_first",
                "synonyms": "set_a",
            },
            "v2": {
                "formality": "formal",
                "phrasing": "imperative",
                "order": "task_first",
                "synonyms": "set_b",
            },
            "v3": {
                "formality": "formal",
                "phrasing": "imperative",
                "order": "text_first",
                "synonyms": "set_a",
            },
        }

        formatted_prompts = {
            "v1": "Analyze the sentiment of the following text and classify it as Very Negative, Negative, Neutral, Positive, or Very Positive...",
            "v2": "Evaluate the emotion of the following text and categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive...",
            "v3": "Text: Great movie!\n\nAnalyze the sentiment of the above text...",
        }

        raw_responses = {
            "v1": ModelResponse(
                content='{"sentiment": "Positive"}',
                model_name="gpt-4.1",
                prompt_used="test",
            ),
            "v2": ModelResponse(
                content='{"sentiment": "Very Positive"}',
                model_name="gpt-4.1",
                prompt_used="test",
            ),
            "v3": ModelResponse(
                content='{"sentiment": "Positive"}',
                model_name="gpt-4.1",
                prompt_used="test",
            ),
        }

        prediction_distribution = {
            "Very Negative": 0.0,
            "Negative": 0.0,
            "Neutral": 0.0,
            "Positive": 0.67,  # 2/3 variants
            "Very Positive": 0.33,  # 1/3 variants
        }

        data = InputConsistencyData(
            input_id=0,
            input_text="Great movie!",
            true_label="Positive",
            variant_predictions=variant_predictions,
            variant_dimensions=variant_dimensions,
            prediction_distribution=prediction_distribution,
            consistency_score=0.67,  # max(prediction_distribution.values())
            majority_label="Positive",
            formatted_prompts=formatted_prompts,
            raw_responses=raw_responses,
        )

        assert data.input_id == 0
        assert data.input_text == "Great movie!"
        assert data.true_label == "Positive"
        assert data.variant_predictions == variant_predictions
        assert data.variant_dimensions == variant_dimensions
        assert data.prediction_distribution == prediction_distribution
        assert data.consistency_score == 0.67
        assert data.majority_label == "Positive"
        assert data.formatted_prompts == formatted_prompts
        assert data.raw_responses == raw_responses

    def test_input_consistency_data_prediction_distribution_calculation(self):
        """Test that prediction distribution calculation is correct."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            InputConsistencyData,
        )

        # Test case: 12/16 variants predict "Positive", 4/16 predict "Very Positive"
        variant_predictions = {}
        for i in range(12):
            variant_predictions[f"v{i + 1}"] = "Positive"
        for i in range(4):
            variant_predictions[f"v{i + 13}"] = "Very Positive"

        expected_distribution = {
            "Very Negative": 0.0,
            "Negative": 0.0,
            "Neutral": 0.0,
            "Positive": 0.75,  # 12/16
            "Very Positive": 0.25,  # 4/16
        }

        # This should be calculated automatically by a helper method
        calculated_distribution = (
            InputConsistencyData.calculate_prediction_distribution(variant_predictions)
        )

        assert calculated_distribution == expected_distribution
        assert max(calculated_distribution.values()) == 0.75  # Consistency score


class TestModelConsistencyData:
    """Test the ModelConsistencyData structure for capturing model-level analysis."""

    def test_model_consistency_data_creation(self):
        """Test creating ModelConsistencyData with aggregated information."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            ModelConsistencyData,
            InputConsistencyData,
        )

        # Mock per-input consistency data
        input_consistency_data = [
            Mock(spec=InputConsistencyData, consistency_score=0.75),
            Mock(spec=InputConsistencyData, consistency_score=0.50),
            Mock(spec=InputConsistencyData, consistency_score=1.0),
        ]

        per_input_consistency = [0.75, 0.50, 1.0]
        overall_consistency = sum(per_input_consistency) / len(
            per_input_consistency
        )  # 0.75

        dimensional_consistency = {
            "formality": 0.72,
            "phrasing": 0.65,
            "order": 0.70,
            "synonyms": 0.66,
        }

        data = ModelConsistencyData(
            model_name="gpt-4.1",
            overall_consistency=overall_consistency,
            per_input_consistency=per_input_consistency,
            input_consistency_data=input_consistency_data,
            dimensional_consistency=dimensional_consistency,
            total_inputs=3,
            total_variants=16,
        )

        assert data.model_name == "gpt-4.1"
        assert data.overall_consistency == 0.75
        assert data.per_input_consistency == per_input_consistency
        assert data.input_consistency_data == input_consistency_data
        assert data.dimensional_consistency == dimensional_consistency
        assert data.total_inputs == 3
        assert data.total_variants == 16


class TestEnhancedConsistencyCalculation:
    """Test the corrected consistency calculation logic."""

    def test_per_input_consistency_calculation(self):
        """Test per-input consistency calculation as max(prediction_distribution)."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EnhancedConsistencyMetric,
        )

        metric = EnhancedConsistencyMetric()

        # Test case 1: Perfect consistency (all variants agree)
        variant_predictions_input = ["Positive", "Positive", "Positive", "Positive"]
        consistency = metric.calculate_per_input_consistency(variant_predictions_input)
        assert consistency == 1.0

        # Test case 2: 75% consistency (3/4 variants agree)
        variant_predictions_input = [
            "Positive",
            "Positive",
            "Positive",
            "Very Positive",
        ]
        consistency = metric.calculate_per_input_consistency(variant_predictions_input)
        assert consistency == 0.75

        # Test case 3: No consistency (all different, max is 25%)
        variant_predictions_input = ["Very Negative", "Negative", "Neutral", "Positive"]
        consistency = metric.calculate_per_input_consistency(variant_predictions_input)
        assert consistency == 0.25

    def test_model_consistency_calculation(self):
        """Test model consistency as mean of per-input consistency scores."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EnhancedConsistencyMetric,
        )

        metric = EnhancedConsistencyMetric()

        # Mock per-input consistency scores
        per_input_scores = [0.75, 0.50, 1.0, 0.25]
        expected_model_consistency = sum(per_input_scores) / len(
            per_input_scores
        )  # 0.625

        model_consistency = metric.calculate_model_consistency(per_input_scores)
        assert model_consistency == expected_model_consistency

    def test_corrected_consistency_workflow(self):
        """Test the complete corrected consistency calculation workflow."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EnhancedConsistencyMetric,
        )

        metric = EnhancedConsistencyMetric()

        # Test data: 3 inputs, 4 variants each
        variant_predictions = {
            "v1": [
                "Positive",
                "Negative",
                "Positive",
            ],  # Input predictions across variants
            "v2": ["Positive", "Positive", "Positive"],
            "v3": ["Very Positive", "Negative", "Positive"],
            "v4": ["Positive", "Neutral", "Very Positive"],
        }

        # Expected per-input consistency:
        # Input 0: ["Positive", "Positive", "Very Positive", "Positive"] -> 3/4 = 0.75
        # Input 1: ["Negative", "Positive", "Negative", "Neutral"] -> 2/4 = 0.50
        # Input 2: ["Positive", "Positive", "Positive", "Very Positive"] -> 3/4 = 0.75

        expected_per_input = [0.75, 0.50, 0.75]
        expected_model_consistency = sum(expected_per_input) / len(
            expected_per_input
        )  # 0.67

        model_consistency_data = metric.calculate_enhanced_consistency(
            variant_predictions
        )

        assert len(model_consistency_data.per_input_consistency) == 3
        assert model_consistency_data.per_input_consistency == expected_per_input
        assert (
            abs(model_consistency_data.overall_consistency - expected_model_consistency)
            < 1e-10
        )


class TestEnhancedEvaluationResult:
    """Test enhanced evaluation result with complete data persistence."""

    def test_enhanced_evaluation_result_creation(self):
        """Test creating enhanced evaluation result with all required data."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EnhancedEvaluationResult,
        )

        # This should extend the existing EvaluationResult with additional fields
        result = EnhancedEvaluationResult(
            model_name="gpt-4.1",
            prompt_variant_id="v1",
            variant_dimensions={
                "formality": "formal",
                "phrasing": "imperative",
                "order": "task_first",
                "synonyms": "set_a",
            },
            variant_name="Formal + Imperative + Task-first + Synonym A",
            custom_accuracy=0.90,
            consistency_score=0.68,  # Model-level consistency
            weighted_index=0.834,
            predictions=["Positive", "Negative", "Neutral"],
            true_labels=["Positive", "Neutral", "Neutral"],
            error_breakdown={"correct": 2, "adjacent": 1},
            formatted_prompts=["prompt1", "prompt2", "prompt3"],
            raw_responses=[Mock(), Mock(), Mock()],
            execution_metadata={
                "avg_response_time": 1.2,
                "total_tokens_used": 1500,
                "error_count": 0,
            },
        )

        assert result.model_name == "gpt-4.1"
        assert result.variant_dimensions["formality"] == "formal"
        assert result.variant_name == "Formal + Imperative + Task-first + Synonym A"
        assert result.formatted_prompts == ["prompt1", "prompt2", "prompt3"]
        assert len(result.raw_responses) == 3
        assert result.execution_metadata["avg_response_time"] == 1.2


class TestDataPersistenceValidation:
    """Test that all required data is properly captured and can be persisted."""

    def test_complete_audit_trail_capture(self):
        """Test that we capture a complete audit trail for analytical notebook."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            InputConsistencyData,
            ModelConsistencyData,
            EnhancedEvaluationResult,
        )

        # Simulate a complete evaluation run with all data captured
        test_inputs = ["Great movie!", "Terrible film", "It's okay"]
        true_labels = ["Positive", "Negative", "Neutral"]

        # For each input, we should capture all variant data
        input_data_list = []
        for i, (text, label) in enumerate(zip(test_inputs, true_labels)):
            variant_predictions = {
                "v1": "Positive",
                "v2": "Positive",
                "v3": "Very Positive",
            }
            variant_dimensions = {
                "v1": {
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_a",
                },
                "v2": {
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_b",
                },
                "v3": {
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "text_first",
                    "synonyms": "set_a",
                },
            }

            input_data = InputConsistencyData(
                input_id=i,
                input_text=text,
                true_label=label,
                variant_predictions=variant_predictions,
                variant_dimensions=variant_dimensions,
                prediction_distribution={
                    "Positive": 0.67,
                    "Very Positive": 0.33,
                    "Negative": 0.0,
                    "Neutral": 0.0,
                    "Very Negative": 0.0,
                },
                consistency_score=0.67,
                majority_label="Positive",
                formatted_prompts={
                    f"v{j + 1}": f"prompt_{j + 1}_for_input_{i}" for j in range(3)
                },
                raw_responses={f"v{j + 1}": Mock(spec=ModelResponse) for j in range(3)},
            )
            input_data_list.append(input_data)

        # Validate that all required data is present
        for input_data in input_data_list:
            # Test-set inputs ✅
            assert input_data.input_text is not None
            assert input_data.true_label is not None

            # Prompt variant dimensions ✅
            for variant_id, dimensions in input_data.variant_dimensions.items():
                assert "formality" in dimensions
                assert "phrasing" in dimensions
                assert "order" in dimensions
                assert "synonyms" in dimensions

            # LLM call outputs ✅
            assert len(input_data.formatted_prompts) == 3
            assert len(input_data.raw_responses) == 3

            # Evaluated metrics ✅
            assert input_data.consistency_score is not None
            assert input_data.prediction_distribution is not None
            assert (
                sum(input_data.prediction_distribution.values()) == 1.0
            )  # Valid probability distribution

    def test_serialization_compatibility(self):
        """Test that enhanced data structures can be serialized for persistence."""
        import json
        from dataclasses import asdict
        from deep_learning_final_assignment.core.evaluation.metrics import (
            InputConsistencyData,
        )

        # Create test data
        input_data = InputConsistencyData(
            input_id=0,
            input_text="Test text",
            true_label="Positive",
            variant_predictions={"v1": "Positive"},
            variant_dimensions={"v1": {"formality": "formal"}},
            prediction_distribution={
                "Positive": 1.0,
                "Negative": 0.0,
                "Neutral": 0.0,
                "Very Positive": 0.0,
                "Very Negative": 0.0,
            },
            consistency_score=1.0,
            majority_label="Positive",
            formatted_prompts={"v1": "test prompt"},
            raw_responses={
                "v1": Mock(
                    spec=ModelResponse,
                    content="test",
                    model_name="test",
                    prompt_used="test",
                )
            },
        )

        # Should be able to convert to dict for JSON serialization
        # Note: raw_responses will need special handling due to ModelResponse objects
        data_dict = asdict(input_data)
        assert data_dict["input_id"] == 0
        assert data_dict["variant_predictions"]["v1"] == "Positive"

        # Should be able to serialize most fields (excluding complex objects)
        serializable_dict = {k: v for k, v in data_dict.items() if k != "raw_responses"}
        json_str = json.dumps(serializable_dict)
        assert "input_text" in json_str
        assert "variant_dimensions" in json_str
