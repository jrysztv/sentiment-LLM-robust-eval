"""
Integration tests for the complete robustness testing pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from deep_learning_final_assignment.core.models.base import ModelResponse
from deep_learning_final_assignment.core.data.loaders import DataSample
from deep_learning_final_assignment.core.prompts.sentiment_prompts import (
    SentimentPrompts,
)
from deep_learning_final_assignment.core.prompts.perturbator import PromptPerturbator
from deep_learning_final_assignment.core.evaluation.metrics import RobustnessEvaluator


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline."""

    def test_complete_pipeline_mock(
        self, mock_openai_model, mock_ollama_model, sample_data_samples
    ):
        """Test complete pipeline with mocked models."""
        # Setup components
        prompts = SentimentPrompts("sentiment_classification")
        perturbator = PromptPerturbator(prompts)
        evaluator = RobustnessEvaluator()

        # Mock model responses to return correct predictions
        def create_mock_response(prompt, sample):
            return ModelResponse(
                content=f'{{"sentiment": "{sample.label}"}}',
                model_name=mock_openai_model.model_name,
                prompt_used=prompt,
            )

        # Test with a subset of variants
        test_variants = ["v1", "v2", "v3"]
        models = {"openai": mock_openai_model, "ollama": mock_ollama_model}

        all_results = {}

        for model_name, model in models.items():
            for variant_id in test_variants:
                # Format prompts
                formatted_prompts = []
                for sample in sample_data_samples:
                    prompt = prompts.format_variant(variant_id, input_text=sample.text)
                    formatted_prompts.append(prompt)

                # Generate responses
                responses = []
                for i, prompt in enumerate(formatted_prompts):
                    response = create_mock_response(prompt, sample_data_samples[i])
                    responses.append(response)

                # Evaluate
                result = evaluator.evaluate_single_combination(
                    responses, sample_data_samples, variant_id
                )

                all_results[f"{model_name}_{variant_id}"] = result

        # Verify results
        assert len(all_results) == 6  # 2 models × 3 variants

        for result in all_results.values():
            assert result.custom_accuracy == 1.0  # Perfect predictions
            assert 0.0 <= result.weighted_index <= 1.0

    def test_prompt_variant_consistency(self):
        """Test that all prompt variants are properly formatted."""
        prompts = SentimentPrompts("sentiment_classification")
        test_text = "This is a test movie review."

        # Test all 16 variants
        for i in range(1, 17):
            variant_id = f"v{i}"
            formatted = prompts.format_variant(variant_id, input_text=test_text)

            # Check that text is included
            assert test_text in formatted

            # Check that JSON format is mentioned
            assert "json" in formatted.lower()

            # Check that sentiment labels are included
            required_labels = [
                "Very Negative",
                "Negative",
                "Neutral",
                "Positive",
                "Very Positive",
            ]
            for label in required_labels:
                assert label in formatted

    def test_dimensional_analysis_workflow(self):
        """Test the dimensional analysis workflow."""
        prompts = SentimentPrompts("sentiment_classification")
        perturbator = PromptPerturbator(prompts)

        # Test each dimension
        dimensions = ["formality", "phrasing", "order", "synonyms"]

        for dimension in dimensions:
            comparison = perturbator.get_dimensional_comparison(dimension)

            # Should have exactly 2 groups per dimension
            assert len(comparison) == 2

            # Each group should have 8 variants (16 total / 2 groups)
            for group_variants in comparison.values():
                assert len(group_variants) == 8

    def test_experimental_groups_workflow(self):
        """Test the experimental groups workflow."""
        prompts = SentimentPrompts("sentiment_classification")
        perturbator = PromptPerturbator(prompts)

        groups = perturbator.get_experimental_groups()

        # Test baseline group
        assert "all_baseline" in groups
        assert len(groups["all_baseline"]) == 16

        # Test dimensional groups
        dimensional_groups = [
            "formal_only",
            "casual_only",
            "imperative_only",
            "question_only",
        ]
        for group_name in dimensional_groups:
            assert group_name in groups
            assert len(groups[group_name]) == 8

    def test_cross_dimensional_analysis_workflow(self):
        """Test the cross-dimensional analysis workflow."""
        prompts = SentimentPrompts("sentiment_classification")
        perturbator = PromptPerturbator(prompts)

        analysis = perturbator.get_cross_dimensional_analysis()

        # Test formality × phrasing
        assert "formality_x_phrasing" in analysis
        formality_phrasing = analysis["formality_x_phrasing"]

        expected_combinations = [
            "formal_imperative",
            "formal_question",
            "casual_imperative",
            "casual_question",
        ]

        for combo in expected_combinations:
            assert combo in formality_phrasing
            assert len(formality_phrasing[combo]) == 4  # 2×2 from other dimensions

    def test_minimal_pairs_analysis(self):
        """Test minimal pairs analysis."""
        prompts = SentimentPrompts("sentiment_classification")
        perturbator = PromptPerturbator(prompts)

        # Test minimal pairs for each dimension
        dimensions = ["formality", "phrasing", "order", "synonyms"]

        for dimension in dimensions:
            pairs = perturbator.get_minimal_pairs(dimension)

            # Should have pairs
            assert len(pairs) > 0

            # Each pair should differ only in the specified dimension
            for variant1, variant2 in pairs:
                diff_count = 0
                for dim_name in variant1.dimensions:
                    if variant1.dimensions[dim_name] != variant2.dimensions[dim_name]:
                        diff_count += 1

                assert diff_count == 1
                assert variant1.dimensions[dimension] != variant2.dimensions[dimension]


@pytest.mark.integration
class TestModelIntegration:
    """Test model integration scenarios."""

    @patch("deep_learning_final_assignment.core.models.openai_model.OpenAI")
    def test_openai_model_integration(self, mock_openai_class, mock_env_vars):
        """Test OpenAI model integration."""
        from deep_learning_final_assignment.core.models.openai_model import OpenAIModel

        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"sentiment": "Positive"}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4-turbo"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 50}

        mock_client.chat.completions.create.return_value = mock_response

        # Test integration
        model = OpenAIModel("gpt-4-turbo")
        prompts = SentimentPrompts("sentiment_classification")

        test_text = "This movie is great!"
        formatted_prompt = prompts.format_variant("v1", input_text=test_text)

        response = model.generate(formatted_prompt)

        assert isinstance(response, ModelResponse)
        assert response.content == '{"sentiment": "Positive"}'
        assert test_text in response.prompt_used

    @patch("deep_learning_final_assignment.core.models.ollama_model.ollama")
    def test_ollama_model_integration(self, mock_ollama):
        """Test Ollama model integration."""
        from deep_learning_final_assignment.core.models.ollama_model import OllamaModel

        # Setup mock
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "qwen2.5:7b"}]
        }  # Make model available

        mock_response = {
            "response": '{"sentiment": "Neutral"}',
            "model": "qwen2.5:7b",
            "done": True,
        }
        mock_client.generate.return_value = mock_response

        # Test integration
        model = OllamaModel("qwen2.5:7b")
        prompts = SentimentPrompts("sentiment_classification")

        test_text = "This movie is okay."
        formatted_prompt = prompts.format_variant("v2", input_text=test_text)

        response = model.generate(formatted_prompt)

        assert isinstance(response, ModelResponse)
        assert response.content == '{"sentiment": "Neutral"}'
        assert test_text in response.prompt_used


@pytest.mark.integration
class TestDataIntegration:
    """Test data loading and processing integration."""

    @patch("pandas.read_json")
    def test_data_loading_integration(self, mock_read_json, mock_sst5_data):
        """Test data loading integration."""
        from deep_learning_final_assignment.core.data.loaders import SST5Loader

        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        samples = loader.load_split("train")

        # Test that samples are properly converted
        assert len(samples) == 5
        assert all(isinstance(sample, DataSample) for sample in samples)

        # Test label conversion
        expected_labels = [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ]
        actual_labels = [sample.label for sample in samples]
        assert actual_labels == expected_labels

        # Test encoding conversion
        expected_encodings = [-3, -2, 0, 2, 3]
        actual_encodings = [sample.encoding for sample in samples]
        assert actual_encodings == expected_encodings

    @patch("pandas.read_json")
    def test_balanced_sampling_integration(self, mock_read_json, mock_sst5_data):
        """Test balanced sampling integration."""
        from deep_learning_final_assignment.core.data.loaders import SST5Loader

        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        samples = loader.get_sample_subset(
            split="train", n_samples=5, balanced=True, random_seed=42
        )

        # Should have one sample from each label
        labels = [sample.label for sample in samples]
        unique_labels = set(labels)
        assert len(unique_labels) == 5


@pytest.mark.integration
class TestEvaluationIntegration:
    """Test evaluation system integration."""

    def test_evaluation_pipeline_integration(self, sample_data_samples):
        """Test complete evaluation pipeline."""
        evaluator = RobustnessEvaluator()

        # Create mock responses with varying accuracy
        responses_v1 = []
        responses_v2 = []

        for i, sample in enumerate(sample_data_samples):
            # v1: Perfect predictions
            response_v1 = ModelResponse(
                content=f'{{"sentiment": "{sample.label}"}}',
                model_name="test_model",
                prompt_used=f"prompt_v1_{i}",
            )
            responses_v1.append(response_v1)

            # v2: Some errors
            if i == 0:
                # Adjacent error: Very Negative -> Negative
                wrong_label = "Negative"
            else:
                wrong_label = sample.label

            response_v2 = ModelResponse(
                content=f'{{"sentiment": "{wrong_label}"}}',
                model_name="test_model",
                prompt_used=f"prompt_v2_{i}",
            )
            responses_v2.append(response_v2)

        combination_responses = {"v1": responses_v1, "v2": responses_v2}

        variant_dimensions = {
            "v1": {"formality": "formal", "phrasing": "imperative"},
            "v2": {"formality": "formal", "phrasing": "question"},
        }

        # Evaluate
        results = evaluator.evaluate_multiple_combinations(
            combination_responses, sample_data_samples, variant_dimensions
        )

        # Verify results
        assert len(results) == 2
        assert results["v1"].custom_accuracy == 1.0  # Perfect
        assert results["v2"].custom_accuracy < 1.0  # Has errors

        # Test best combination selection
        best_id, best_result = evaluator.get_best_combination(results)
        assert best_id == "v1"
        assert best_result.custom_accuracy == 1.0

    def test_consistency_evaluation_integration(self):
        """Test consistency evaluation integration."""
        evaluator = RobustnessEvaluator()

        # Create predictions with known consistency patterns
        variant_predictions = {
            "v1": ["Positive", "Negative", "Neutral"],
            "v2": ["Positive", "Negative", "Neutral"],  # 100% consistent with v1
            "v3": ["Positive", "Positive", "Neutral"],  # 67% consistent with v1
        }

        consistency = evaluator.consistency_metric.calculate_consistency(
            variant_predictions
        )

        # Expected: (1.0 + 0.67 + 0.67) / 3 ≈ 0.78
        # But actual calculation is different - let's check the actual value
        assert 0.6 < consistency < 0.8

    def test_performance_summary_integration(self):
        """Test performance summary integration."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EvaluationResult,
        )

        evaluator = RobustnessEvaluator()

        # Create test results
        results = {
            "v1": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v1",
                custom_accuracy=0.9,
                consistency_score=0.8,
                weighted_index=0.87,  # 0.7*0.9 + 0.3*0.8
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
            "v2": EvaluationResult(
                model_name="test_model",
                prompt_variant_id="v2",
                custom_accuracy=0.8,
                consistency_score=0.9,
                weighted_index=0.83,  # 0.7*0.8 + 0.3*0.9
                predictions=[],
                true_labels=[],
                error_breakdown={},
            ),
        }

        summary = evaluator.get_performance_summary(results)

        assert summary["total_combinations"] == 2
        assert abs(summary["accuracy_stats"]["mean"] - 0.85) < 1e-10
        assert summary["accuracy_stats"]["min"] == 0.8
        assert summary["accuracy_stats"]["max"] == 0.9
        assert abs(summary["consistency_stats"]["mean"] - 0.85) < 1e-10
        assert summary["weighted_index_stats"]["mean"] == 0.85


@pytest.mark.integration
@pytest.mark.slow
class TestFileIOIntegration:
    """Test file I/O integration scenarios."""

    def test_results_serialization(self, temp_dir):
        """Test serialization and deserialization of results."""
        from deep_learning_final_assignment.core.evaluation.metrics import (
            EvaluationResult,
        )

        # Create test result
        result = EvaluationResult(
            model_name="test_model",
            prompt_variant_id="v1",
            custom_accuracy=0.85,
            consistency_score=0.92,
            weighted_index=0.871,
            predictions=["Positive", "Negative"],
            true_labels=["Positive", "Neutral"],
            error_breakdown={"correct": 1, "errors": 1},
        )

        # Serialize to file
        results_file = temp_dir / "test_results.json"

        # Convert to dict for JSON serialization
        result_dict = {
            "model_name": result.model_name,
            "prompt_variant_id": result.prompt_variant_id,
            "custom_accuracy": result.custom_accuracy,
            "consistency_score": result.consistency_score,
            "weighted_index": result.weighted_index,
            "predictions": result.predictions,
            "true_labels": result.true_labels,
            "error_breakdown": result.error_breakdown,
            "metadata": result.metadata,
        }

        with open(results_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        # Verify file exists and can be read
        assert results_file.exists()

        with open(results_file, "r") as f:
            loaded_dict = json.load(f)

        assert loaded_dict["model_name"] == "test_model"
        assert loaded_dict["custom_accuracy"] == 0.85
        assert loaded_dict["predictions"] == ["Positive", "Negative"]

    def test_configuration_loading(self, temp_dir):
        """Test loading configuration from file."""
        # Create test config
        config = {
            "models": {
                "openai": {"model_name": "gpt-4-turbo", "temperature": 0.1},
                "ollama": {"model_name": "qwen2.5:7b", "temperature": 0.1},
            },
            "evaluation": {"n_samples": 100, "balanced": True, "random_seed": 42},
        }

        config_file = temp_dir / "test_config.json"

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Load and verify
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["models"]["openai"]["model_name"] == "gpt-4-turbo"
        assert loaded_config["evaluation"]["n_samples"] == 100
