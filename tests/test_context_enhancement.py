"""
Tests for Phase 2 context enhancement functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List

from deep_learning_final_assignment.core.context_enhancement import (
    ContextSelector,
    ContextEnhancedVariantGenerator,
    Phase2ContextEnhancement,
    ContextExample,
    EnhancedVariant,
)
from deep_learning_final_assignment.core.data import SST5Loader, DataSample
from deep_learning_final_assignment.core.prompts import SentimentPrompts
from deep_learning_final_assignment.core.evaluation.metrics import EvaluationResult


@pytest.fixture
def mock_sst5_loader():
    """Mock SST5Loader for testing."""
    loader = Mock(spec=SST5Loader)

    # Mock training samples
    train_samples = [
        DataSample(text="This movie is terrible", label="Very Negative", encoding=-3),
        DataSample(text="Bad film", label="Negative", encoding=-2),
        DataSample(text="Okay movie", label="Neutral", encoding=0),
        DataSample(text="Good film", label="Positive", encoding=2),
        DataSample(text="Amazing movie experience", label="Very Positive", encoding=3),
        DataSample(text="Absolutely horrible", label="Very Negative", encoding=-3),
        DataSample(text="Not great", label="Negative", encoding=-2),
        DataSample(text="Average film", label="Neutral", encoding=0),
        DataSample(text="Really good", label="Positive", encoding=2),
        DataSample(text="Outstanding masterpiece", label="Very Positive", encoding=3),
    ]

    # Mock validation samples (different from training)
    validation_samples = [
        DataSample(text="Great movie!", label="Positive", encoding=2),
        DataSample(text="Boring film", label="Negative", encoding=-2),
    ]

    loader.load_split.return_value = train_samples
    loader.get_sample_subset.return_value = validation_samples

    return loader


@pytest.fixture
def mock_sentiment_prompts():
    """Mock SentimentPrompts for testing."""
    prompts = Mock(spec=SentimentPrompts)

    # Mock variant
    mock_variant = Mock()
    mock_variant.id = "v1"
    mock_variant.template = "Analyze the sentiment: {input_text}"
    mock_variant.dimensions = {
        "formality": "formal",
        "phrasing": "imperative",
        "order": "task_first",
        "synonyms": "set_a",
    }

    prompts.get_variant.return_value = mock_variant
    return prompts


@pytest.fixture
def sample_baseline_results():
    """Sample baseline results for testing."""
    # Create mock EvaluationResult objects
    result1 = EvaluationResult(
        model_name="gpt-4.1",
        prompt_variant_id="v1",
        custom_accuracy=0.85,
        consistency_score=0.70,
        weighted_index=0.805,  # 0.7*0.85 + 0.3*0.70
        predictions=["Positive", "Negative"],
        true_labels=["Positive", "Negative"],
        error_breakdown={},
    )

    result2 = EvaluationResult(
        model_name="gpt-4.1",
        prompt_variant_id="v2",
        custom_accuracy=0.90,
        consistency_score=0.70,
        weighted_index=0.84,  # 0.7*0.90 + 0.3*0.70
        predictions=["Positive", "Negative"],
        true_labels=["Positive", "Negative"],
        error_breakdown={},
    )

    result3 = EvaluationResult(
        model_name="gpt-4o-mini",
        prompt_variant_id="v1",
        custom_accuracy=0.80,
        consistency_score=0.65,
        weighted_index=0.755,  # 0.7*0.80 + 0.3*0.65
        predictions=["Positive", "Negative"],
        true_labels=["Positive", "Negative"],
        error_breakdown={},
    )

    return {"gpt-4.1": {"v1": result1, "v2": result2}, "gpt-4o-mini": {"v1": result3}}


class TestContextSelector:
    """Test ContextSelector functionality."""

    def test_get_length_category(self, mock_sst5_loader):
        """Test text length categorization."""
        selector = ContextSelector(mock_sst5_loader)

        assert selector._get_length_category("Short text") == "short"
        assert (
            selector._get_length_category(
                "This is a medium length text with several words"
            )
            == "medium"
        )
        assert (
            selector._get_length_category(
                "This is a very long text with many words that should be categorized as long based on word count and it has more than twenty words total"
            )
            == "long"
        )

    def test_select_diverse_examples(self, mock_sst5_loader):
        """Test diverse example selection."""
        selector = ContextSelector(mock_sst5_loader)

        # Mock TF-IDF vectorizer
        with patch(
            "deep_learning_final_assignment.core.context_enhancement.TfidfVectorizer"
        ) as mock_vectorizer:
            mock_vec_instance = Mock()
            mock_vectorizer.return_value = mock_vec_instance
            mock_vec_instance.fit_transform.return_value = Mock()
            mock_vec_instance.transform.return_value = Mock()

            # Mock cosine similarity
            with patch(
                "deep_learning_final_assignment.core.context_enhancement.cosine_similarity"
            ) as mock_cosine:
                mock_cosine.return_value = [[0.5]]  # Medium similarity

                result = selector.select_diverse_examples(n_per_label=2)

                # Should return examples for each label
                assert isinstance(result, dict)
                # Should have examples for available labels
                assert len(result) > 0

                # Check structure of returned examples
                for label, examples in result.items():
                    assert isinstance(examples, list)
                    for example in examples:
                        assert isinstance(example, ContextExample)
                        assert example.label == label
                        assert example.source_split == "train"

    def test_validate_no_contamination_success(self, mock_sst5_loader):
        """Test contamination validation success case."""
        selector = ContextSelector(mock_sst5_loader)

        # Different texts - no contamination
        context_examples = {
            "Positive": [
                ContextExample(
                    text="Training text 1",
                    label="Positive",
                    length_category="short",
                    tfidf_dissimilarity_score=1.0,
                    selection_reason="test",
                )
            ]
        }

        test_samples = [DataSample(text="Test text 1", label="Positive", encoding=2)]

        result = selector.validate_no_contamination(context_examples, test_samples)

        assert result["validation_status"] == "NO_CONTAMINATION_DETECTED"
        assert result["overlapping_examples"] == 0

    def test_validate_no_contamination_failure(self, mock_sst5_loader):
        """Test contamination validation failure case."""
        selector = ContextSelector(mock_sst5_loader)

        # Same text - contamination detected
        same_text = "This is the same text"
        context_examples = {
            "Positive": [
                ContextExample(
                    text=same_text,
                    label="Positive",
                    length_category="short",
                    tfidf_dissimilarity_score=1.0,
                    selection_reason="test",
                )
            ]
        }

        test_samples = [DataSample(text=same_text, label="Positive", encoding=2)]

        with pytest.raises(ValueError, match="CONTAMINATION DETECTED"):
            selector.validate_no_contamination(context_examples, test_samples)


class TestContextEnhancedVariantGenerator:
    """Test ContextEnhancedVariantGenerator functionality."""

    def test_format_context_examples(self, mock_sentiment_prompts):
        """Test context example formatting."""
        generator = ContextEnhancedVariantGenerator(mock_sentiment_prompts)

        examples = {
            "Positive": [
                ContextExample(
                    text="Great movie",
                    label="Positive",
                    length_category="short",
                    tfidf_dissimilarity_score=1.0,
                    selection_reason="test",
                )
            ],
            "Negative": [
                ContextExample(
                    text="Bad film",
                    label="Negative",
                    length_category="short",
                    tfidf_dissimilarity_score=1.0,
                    selection_reason="test",
                )
            ],
        }

        formatted = generator._format_context_examples(examples)

        assert "Here are examples of sentiment classification:" in formatted
        assert "Positive: 'Great movie'" in formatted
        assert "Negative: 'Bad film'" in formatted

    def test_generate_enhanced_variants(self, mock_sentiment_prompts):
        """Test enhanced variant generation."""
        generator = ContextEnhancedVariantGenerator(mock_sentiment_prompts)

        examples = {
            "Positive": [
                ContextExample(
                    text="Great movie",
                    label="Positive",
                    length_category="short",
                    tfidf_dissimilarity_score=1.0,
                    selection_reason="test",
                )
            ]
        }

        prefix_variant, suffix_variant = generator.generate_enhanced_variants(
            "v1", examples
        )

        # Check prefix variant
        assert isinstance(prefix_variant, EnhancedVariant)
        assert prefix_variant.variant_id == "v1_prefix"
        assert prefix_variant.context_position == "prefix"
        assert prefix_variant.base_variant == "v1"
        assert "Here are examples" in prefix_variant.full_template
        assert "Analyze the sentiment" in prefix_variant.full_template

        # Check suffix variant
        assert isinstance(suffix_variant, EnhancedVariant)
        assert suffix_variant.variant_id == "v1_suffix"
        assert suffix_variant.context_position == "suffix"
        assert suffix_variant.base_variant == "v1"
        assert "Here are examples" in suffix_variant.full_template
        assert "Analyze the sentiment" in suffix_variant.full_template


class TestPhase2ContextEnhancement:
    """Test complete Phase 2 system."""

    def test_run_context_enhancement(
        self, mock_sst5_loader, mock_sentiment_prompts, sample_baseline_results
    ):
        """Test complete context enhancement process."""
        # Mock test samples
        test_samples = [
            DataSample(text="Test movie", label="Positive", encoding=2),
            DataSample(text="Bad test film", label="Negative", encoding=-2),
        ]

        phase2_system = Phase2ContextEnhancement(
            mock_sst5_loader, mock_sentiment_prompts
        )

        # Mock the context selector methods
        with patch.object(
            phase2_system.context_selector, "select_diverse_examples"
        ) as mock_select:
            with patch.object(
                phase2_system.context_selector, "validate_no_contamination"
            ) as mock_validate:
                # Mock return values
                mock_context_examples = {
                    "Positive": [
                        ContextExample(
                            text="Training positive",
                            label="Positive",
                            length_category="short",
                            tfidf_dissimilarity_score=1.0,
                            selection_reason="test",
                        )
                    ]
                }
                mock_select.return_value = mock_context_examples
                mock_validate.return_value = {
                    "validation_status": "NO_CONTAMINATION_DETECTED",
                    "overlapping_examples": 0,
                }

                # Run context enhancement
                result = phase2_system.run_context_enhancement(
                    baseline_results=sample_baseline_results,
                    test_samples=test_samples,
                    n_context_per_label=1,
                    random_seed=42,
                )

                # Verify structure
                assert "audit" in result
                assert "best_baseline_combinations" in result
                assert "enhanced_variants" in result
                assert "context_examples" in result
                assert "validation_results" in result

                # Check best combinations per model
                best_combinations = result["best_baseline_combinations"]
                assert "gpt-4.1" in best_combinations
                assert "gpt-4o-mini" in best_combinations

                # Check enhanced variants
                enhanced_variants = result["enhanced_variants"]
                assert "gpt-4.1" in enhanced_variants
                assert "gpt-4o-mini" in enhanced_variants

                for model_variants in enhanced_variants.values():
                    assert "base_variant" in model_variants
                    assert "prefix_variant" in model_variants
                    assert "suffix_variant" in model_variants

    def test_save_phase2_results(
        self, mock_sst5_loader, mock_sentiment_prompts, tmp_path
    ):
        """Test Phase 2 results saving."""
        phase2_system = Phase2ContextEnhancement(
            mock_sst5_loader, mock_sentiment_prompts
        )

        # Mock phase2 results
        from deep_learning_final_assignment.core.context_enhancement import (
            ContextSelectionAudit,
        )

        mock_audit = ContextSelectionAudit(
            timestamp="2025-01-25T15:30:00Z",
            selection_strategy="test",
            data_split_used="train",
            total_train_samples=100,
            selected_examples_per_label=3,
            total_context_examples=15,
            validation_split_used="validation",
            test_samples=50,
            contamination_check="PASSED",
            selected_context_examples={},
            contamination_validation={"validation_status": "PASSED"},
            tfidf_analysis={},
        )

        mock_results = {
            "audit": mock_audit,
            "best_baseline_combinations": {},
            "enhanced_variants": {},
            "context_examples": {},
            "validation_results": {"validation_status": "PASSED"},
        }

        saved_files = phase2_system.save_phase2_results(mock_results, str(tmp_path))

        # Check files were created
        assert "audit" in saved_files
        assert "variants" in saved_files

        # Verify files exist
        audit_file = Path(saved_files["audit"])
        variants_file = Path(saved_files["variants"])

        assert audit_file.exists()
        assert variants_file.exists()

        # Verify file contents are valid JSON
        with open(audit_file) as f:
            audit_data = json.load(f)
            assert "context_selection_metadata" in audit_data

        with open(variants_file) as f:
            variants_data = json.load(f)
            assert "enhanced_variants_metadata" in variants_data


if __name__ == "__main__":
    pytest.main([__file__])
