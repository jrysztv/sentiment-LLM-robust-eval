"""
Tests for Phase 1.5 data persistence layer.

This module tests the complete data persistence functionality including:
- detailed_execution_data_TIMESTAMP.json (complete audit trail)
- baseline_results_detailed_TIMESTAMP.json (enhanced with dimensions/metadata)
- baseline_results_summary_TIMESTAMP.json (model performance summaries)
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime
from pathlib import Path

from deep_learning_final_assignment.core.models.base import ModelResponse
from deep_learning_final_assignment.core.evaluation.metrics import (
    InputConsistencyData,
    ModelConsistencyData,
    EnhancedEvaluationResult,
)


class TestDataPersistenceManager:
    """Test the DataPersistenceManager for Phase 1.5 enhanced file output."""

    def test_data_persistence_manager_creation(self):
        """Test creating DataPersistenceManager with proper configuration."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        manager = DataPersistenceManager(output_dir="test_output")

        assert manager.output_dir == Path("test_output")
        assert manager.timestamp is not None

    def test_generate_timestamps(self):
        """Test timestamp generation for consistent file naming."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        manager = DataPersistenceManager()

        # Should generate consistent timestamp strings
        timestamp1 = manager.get_timestamp_string()
        timestamp2 = manager.get_timestamp_string()

        assert (
            timestamp1 == timestamp2
        )  # Should be consistent within same manager instance
        assert len(timestamp1) > 0
        assert "_" in timestamp1  # YYYYMMDD_HHMMSS format

    def test_save_detailed_execution_data(self):
        """Test saving detailed_execution_data_TIMESTAMP.json with complete audit trail."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataPersistenceManager(output_dir=temp_dir)

            # Create test data matching the Phase 1.5 specification
            experiment_metadata = {
                "timestamp": "2025-01-25T14:30:00Z",
                "models_evaluated": ["gpt-4.1", "gpt-4o-mini"],
                "total_combinations": 32,
                "total_inputs": 50,
                "configuration": {"test": "config"},
            }

            # Create mock model consistency data
            model_consistency_data = {
                "gpt-4.1": ModelConsistencyData(
                    model_name="gpt-4.1",
                    overall_consistency=0.68,
                    per_input_consistency=[0.75, 0.50, 1.0],
                    input_consistency_data=[],  # Would be populated with real data
                    dimensional_consistency={
                        "formality": 0.72,
                        "phrasing": 0.65,
                        "order": 0.70,
                        "synonyms": 0.66,
                    },
                    total_inputs=3,
                    total_variants=16,
                )
            }

            file_path = manager.save_detailed_execution_data(
                experiment_metadata=experiment_metadata,
                model_consistency_data=model_consistency_data,
            )

            # Verify file was created
            assert file_path.exists()
            assert file_path.name.startswith("detailed_execution_data_")
            assert file_path.suffix == ".json"

            # Verify content structure
            with open(file_path, "r") as f:
                data = json.load(f)

            assert "experiment_metadata" in data
            assert "gpt-4.1" in data
            assert data["experiment_metadata"]["models_evaluated"] == [
                "gpt-4.1",
                "gpt-4o-mini",
            ]
            assert data["gpt-4.1"]["model_summary"]["overall_consistency"] == 0.68

    def test_save_enhanced_detailed_results(self):
        """Test saving baseline_results_detailed_TIMESTAMP.json with enhanced data."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataPersistenceManager(output_dir=temp_dir)

            # Create mock enhanced evaluation results
            enhanced_results = {
                "gpt-4.1": {
                    "v1": EnhancedEvaluationResult(
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
                        consistency_score=0.68,
                        weighted_index=0.834,
                        predictions=["Positive", "Negative", "Neutral"],
                        true_labels=["Positive", "Neutral", "Neutral"],
                        error_breakdown={"correct": 2, "adjacent": 1},
                        formatted_prompts=["prompt1", "prompt2", "prompt3"],
                        raw_responses=[
                            ModelResponse(
                                content="response1",
                                model_name="gpt-4.1",
                                prompt_used="prompt1",
                            ),
                            ModelResponse(
                                content="response2",
                                model_name="gpt-4.1",
                                prompt_used="prompt2",
                            ),
                            ModelResponse(
                                content="response3",
                                model_name="gpt-4.1",
                                prompt_used="prompt3",
                            ),
                        ],
                        execution_metadata={
                            "avg_response_time": 1.2,
                            "total_tokens_used": 1500,
                            "error_count": 0,
                        },
                    )
                }
            }

            file_path = manager.save_enhanced_detailed_results(enhanced_results)

            # Verify file was created
            assert file_path.exists()
            assert file_path.name.startswith("baseline_results_detailed_")
            assert file_path.suffix == ".json"

            # Verify content structure
            with open(file_path, "r") as f:
                data = json.load(f)

            assert "gpt-4.1" in data
            assert "v1" in data["gpt-4.1"]

            variant_data = data["gpt-4.1"]["v1"]
            assert variant_data["variant_dimensions"]["formality"] == "formal"
            assert (
                variant_data["variant_name"]
                == "Formal + Imperative + Task-first + Synonym A"
            )
            assert variant_data["custom_accuracy"] == 0.90
            assert variant_data["execution_metadata"]["avg_response_time"] == 1.2
            assert len(variant_data["raw_responses"]) == 3

    def test_save_enhanced_summary_results(self):
        """Test saving baseline_results_summary_TIMESTAMP.json with performance summaries."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataPersistenceManager(output_dir=temp_dir)

            # Create mock summary data
            summary_results = {
                "gpt-4.1": {
                    "performance_summary": {
                        "total_combinations": 16,
                        "accuracy_stats": {
                            "mean": 0.85,
                            "std": 0.05,
                            "min": 0.80,
                            "max": 0.90,
                        },
                        "consistency_stats": {
                            "mean": 0.68,
                            "std": 0.0,  # Same for all variants
                            "min": 0.68,
                            "max": 0.68,
                        },
                        "weighted_index_stats": {
                            "mean": 0.799,
                            "std": 0.035,
                            "min": 0.764,
                            "max": 0.834,
                        },
                    },
                    "best_combination": {
                        "variant_id": "v11",
                        "custom_accuracy": 0.92,
                        "consistency_score": 0.68,
                        "weighted_index": 0.848,
                    },
                    "dimensional_analysis": {
                        "formality": {"formal": 0.85, "casual": 0.87},
                        "phrasing": {"imperative": 0.86, "question": 0.86},
                        "order": {"task_first": 0.85, "text_first": 0.87},
                        "synonyms": {"set_a": 0.86, "set_b": 0.86},
                    },
                    "total_variants_tested": 16,
                }
            }

            file_path = manager.save_enhanced_summary_results(summary_results)

            # Verify file was created
            assert file_path.exists()
            assert file_path.name.startswith("baseline_results_summary_")
            assert file_path.suffix == ".json"

            # Verify content structure
            with open(file_path, "r") as f:
                data = json.load(f)

            assert "gpt-4.1" in data
            assert data["gpt-4.1"]["performance_summary"]["total_combinations"] == 16
            assert data["gpt-4.1"]["best_combination"]["variant_id"] == "v11"
            assert (
                data["gpt-4.1"]["dimensional_analysis"]["formality"]["formal"] == 0.85
            )

    def test_complete_data_persistence_workflow(self):
        """Test the complete data persistence workflow for Phase 1.5."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataPersistenceManager(output_dir=temp_dir)

            # Simulate complete Phase 1.5 execution data
            experiment_metadata = {
                "timestamp": manager.get_timestamp_string(),
                "models_evaluated": ["gpt-4.1", "gpt-4o-mini"],
                "total_combinations": 32,
                "total_inputs": 50,
            }

            # Create input consistency data for verification
            input_data = InputConsistencyData(
                input_id=0,
                input_text="Great movie!",
                true_label="Positive",
                variant_predictions={"v1": "Positive", "v2": "Very Positive"},
                variant_dimensions={
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
                },
                prediction_distribution={
                    "Positive": 0.5,
                    "Very Positive": 0.5,
                    "Negative": 0.0,
                    "Neutral": 0.0,
                    "Very Negative": 0.0,
                },
                consistency_score=0.5,
                majority_label="Positive",  # Tie, but alphabetically first
                formatted_prompts={"v1": "prompt1", "v2": "prompt2"},
                raw_responses={
                    "v1": ModelResponse(
                        content="response1",
                        model_name="gpt-4.1",
                        prompt_used="prompt1",
                    ),
                    "v2": ModelResponse(
                        content="response2",
                        model_name="gpt-4.1",
                        prompt_used="prompt2",
                    ),
                },
            )

            model_consistency_data = {
                "gpt-4.1": ModelConsistencyData(
                    model_name="gpt-4.1",
                    overall_consistency=0.68,
                    per_input_consistency=[0.5],
                    input_consistency_data=[input_data],
                    dimensional_consistency={
                        "formality": 0.72,
                        "phrasing": 0.65,
                        "order": 0.70,
                        "synonyms": 0.66,
                    },
                    total_inputs=1,
                    total_variants=2,
                )
            }

            # Save all three file types
            detailed_file = manager.save_detailed_execution_data(
                experiment_metadata, model_consistency_data
            )

            # Verify all files were created with consistent timestamps
            assert detailed_file.exists()

            # Verify timestamp consistency in file names
            timestamp = manager.get_timestamp_string()
            assert timestamp in detailed_file.name

            # Verify content integrity across files
            with open(detailed_file, "r") as f:
                detailed_data = json.load(f)

            # Validate complete audit trail is present
            assert "experiment_metadata" in detailed_data
            assert "gpt-4.1" in detailed_data
            assert len(detailed_data["gpt-4.1"]["per_input_analysis"]) == 1

            input_analysis = detailed_data["gpt-4.1"]["per_input_analysis"][0]
            assert input_analysis["input_text"] == "Great movie!"
            assert "variant_predictions" in input_analysis
            assert "variant_dimensions" in input_analysis
            assert "formatted_prompts" in input_analysis
            assert "raw_responses" in input_analysis


class TestDataIntegrityValidation:
    """Test data integrity validation for Phase 1.5 persistence."""

    def test_variant_dimensions_preservation(self):
        """Test that variant dimensions from sentiment_prompts.py are preserved correctly."""
        from deep_learning_final_assignment.core.evaluation.data_persistence import (
            DataPersistenceManager,
        )

        # Test that all four dimensions are preserved
        test_dimensions = {
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
            "v16": {
                "formality": "casual",
                "phrasing": "question",
                "order": "text_first",
                "synonyms": "set_b",
            },
        }

        input_data = InputConsistencyData(
            input_id=0,
            input_text="Test",
            true_label="Positive",
            variant_predictions={"v1": "Positive"},
            variant_dimensions=test_dimensions,
            prediction_distribution={
                "Positive": 1.0,
                "Negative": 0.0,
                "Neutral": 0.0,
                "Very Positive": 0.0,
                "Very Negative": 0.0,
            },
            consistency_score=1.0,
            majority_label="Positive",
            formatted_prompts={"v1": "prompt"},
            raw_responses={
                "v1": ModelResponse(
                    content="test", model_name="test", prompt_used="test"
                )
            },
        )

        serialized = input_data.to_serializable_dict()

        # Verify all dimension information is preserved
        assert serialized["variant_dimensions"]["v1"]["formality"] == "formal"
        assert serialized["variant_dimensions"]["v1"]["phrasing"] == "imperative"
        assert serialized["variant_dimensions"]["v1"]["order"] == "task_first"
        assert serialized["variant_dimensions"]["v1"]["synonyms"] == "set_a"

        assert serialized["variant_dimensions"]["v16"]["formality"] == "casual"
        assert serialized["variant_dimensions"]["v16"]["synonyms"] == "set_b"

    def test_llm_output_preservation(self):
        """Test that complete LLM outputs are preserved with metadata."""
        mock_response = Mock(spec=ModelResponse)
        mock_response.content = '{"sentiment": "Positive"}'
        mock_response.model_name = "gpt-4.1"
        mock_response.prompt_used = "Analyze the sentiment..."
        mock_response.metadata = {"tokens": 25, "response_time": 1.2}

        input_data = InputConsistencyData(
            input_id=0,
            input_text="Test",
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
            formatted_prompts={"v1": "Analyze the sentiment..."},
            raw_responses={"v1": mock_response},
        )

        serialized = input_data.to_serializable_dict()

        # Verify LLM output preservation
        raw_response = serialized["raw_responses"]["v1"]
        assert raw_response["content"] == '{"sentiment": "Positive"}'
        assert raw_response["model_name"] == "gpt-4.1"
        assert raw_response["prompt_used"] == "Analyze the sentiment..."
        assert raw_response["metadata"]["tokens"] == 25
        assert raw_response["metadata"]["response_time"] == 1.2

    def test_evaluation_metrics_preservation(self):
        """Test that all evaluation metrics are properly preserved."""
        result = EnhancedEvaluationResult(
            model_name="gpt-4.1",
            prompt_variant_id="v1",
            variant_dimensions={"formality": "formal"},
            variant_name="Test Variant",
            custom_accuracy=0.85,
            consistency_score=0.68,
            weighted_index=0.799,
            predictions=["Positive", "Negative"],
            true_labels=["Positive", "Neutral"],
            error_breakdown={"correct": 1, "base_neutral": 1},
            execution_metadata={
                "avg_response_time": 1.5,
                "total_tokens_used": 2000,
                "error_count": 0,
            },
        )

        serialized = result.to_serializable_dict()

        # Verify all metrics are preserved
        assert serialized["custom_accuracy"] == 0.85
        assert serialized["consistency_score"] == 0.68
        assert serialized["weighted_index"] == 0.799
        assert serialized["error_breakdown"]["correct"] == 1
        assert serialized["execution_metadata"]["avg_response_time"] == 1.5
        assert serialized["variant_dimensions"]["formality"] == "formal"

    def test_backward_compatibility(self):
        """Test that enhanced results maintain backward compatibility with existing analysis tools."""
        # Enhanced results should include all fields from original EvaluationResult
        result = EnhancedEvaluationResult(
            model_name="gpt-4.1",
            prompt_variant_id="v1",
            custom_accuracy=0.85,
            consistency_score=0.68,
            weighted_index=0.799,
            predictions=["Positive"],
            true_labels=["Positive"],
            error_breakdown={"correct": 1},
        )

        # Should have all original EvaluationResult fields
        assert hasattr(result, "model_name")
        assert hasattr(result, "prompt_variant_id")
        assert hasattr(result, "custom_accuracy")
        assert hasattr(result, "consistency_score")
        assert hasattr(result, "weighted_index")
        assert hasattr(result, "predictions")
        assert hasattr(result, "true_labels")
        assert hasattr(result, "error_breakdown")
        assert hasattr(result, "metadata")

        # Should have enhanced fields
        assert hasattr(result, "variant_dimensions")
        assert hasattr(result, "variant_name")
        assert hasattr(result, "formatted_prompts")
        assert hasattr(result, "raw_responses")
        assert hasattr(result, "execution_metadata")
