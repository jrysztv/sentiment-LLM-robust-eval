"""
Tests for prompt template and perturbation components.
"""

import pytest
from deep_learning_final_assignment.core.prompts.template import (
    PromptVariant,
    PromptTemplate,
)
from deep_learning_final_assignment.core.prompts.sentiment_prompts import (
    SentimentPrompts,
)
from deep_learning_final_assignment.core.prompts.perturbator import PromptPerturbator


class TestPromptVariant:
    """Test the PromptVariant dataclass."""

    def test_prompt_variant_creation(self):
        """Test creating a PromptVariant."""
        variant = PromptVariant(
            id="test_v1",
            name="Test Variant 1",
            template="Analyze this text: {input_text}",
            dimensions={"formality": "formal", "phrasing": "imperative"},
            description="A test variant",
        )

        assert variant.id == "test_v1"
        assert variant.name == "Test Variant 1"
        assert variant.template == "Analyze this text: {input_text}"
        assert variant.dimensions == {"formality": "formal", "phrasing": "imperative"}
        assert variant.description == "A test variant"

    def test_prompt_variant_format(self):
        """Test formatting a prompt variant."""
        variant = PromptVariant(
            id="test_v1",
            name="Test Variant 1",
            template="Analyze this text: {input_text}. The task is {task}.",
            dimensions={"formality": "formal"},
        )

        formatted = variant.format(input_text="Hello world", task="sentiment analysis")
        expected = "Analyze this text: Hello world. The task is sentiment analysis."

        assert formatted == expected

    def test_prompt_variant_default_description(self):
        """Test PromptVariant with default description."""
        variant = PromptVariant(
            id="test_v1", name="Test Variant 1", template="Test template", dimensions={}
        )

        assert variant.description is None


class TestSentimentPrompts:
    """Test the SentimentPrompts class."""

    def test_sentiment_prompts_initialization(self):
        """Test SentimentPrompts initialization."""
        prompts = SentimentPrompts("test_task")

        assert prompts.task_name == "test_task"
        assert len(prompts.variants) == 16  # Should have all 16 variants

    def test_all_variants_present(self):
        """Test that all 16 variants are present."""
        prompts = SentimentPrompts("test_task")

        expected_ids = [f"v{i}" for i in range(1, 17)]
        actual_ids = list(prompts.variants.keys())

        assert set(actual_ids) == set(expected_ids)

    def test_variant_dimensions(self):
        """Test that variants have correct dimensions."""
        prompts = SentimentPrompts("test_task")

        # Test a few specific variants
        v1 = prompts.get_variant("v1")
        assert v1.dimensions == {
            "formality": "formal",
            "phrasing": "imperative",
            "order": "task_first",
            "synonyms": "set_a",
        }

        v16 = prompts.get_variant("v16")
        assert v16.dimensions == {
            "formality": "casual",
            "phrasing": "question",
            "order": "text_first",
            "synonyms": "set_b",
        }

    def test_variant_templates_contain_placeholder(self):
        """Test that all variant templates contain the input placeholder."""
        prompts = SentimentPrompts("test_task")

        for variant in prompts.get_all_variants():
            assert "{input_text}" in variant.template

    def test_variant_templates_contain_json_format(self):
        """Test that all variant templates mention JSON format."""
        prompts = SentimentPrompts("test_task")

        for variant in prompts.get_all_variants():
            template_lower = variant.template.lower()
            assert "json" in template_lower

    def test_variant_templates_contain_sentiment_labels(self):
        """Test that all variant templates contain the sentiment labels."""
        prompts = SentimentPrompts("test_task")

        required_labels = [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ]

        for variant in prompts.get_all_variants():
            for label in required_labels:
                assert label in variant.template

    def test_format_variant(self):
        """Test formatting a specific variant."""
        prompts = SentimentPrompts("test_task")

        formatted = prompts.format_variant("v1", input_text="This is a test movie.")

        assert "This is a test movie." in formatted
        assert "Analyze the sentiment" in formatted
        assert "JSON format" in formatted

    def test_format_all_variants(self):
        """Test formatting all variants."""
        prompts = SentimentPrompts("test_task")

        formatted_all = prompts.format_all_variants(input_text="Test text")

        assert len(formatted_all) == 16
        assert all("Test text" in prompt for prompt in formatted_all.values())

    def test_get_variants_by_dimension(self):
        """Test getting variants by dimension."""
        prompts = SentimentPrompts("test_task")

        formal_variants = prompts.get_variants_by_dimension("formality", "formal")
        casual_variants = prompts.get_variants_by_dimension("formality", "casual")

        assert len(formal_variants) == 8  # Half should be formal
        assert len(casual_variants) == 8  # Half should be casual

        # Check that all formal variants have formal dimension
        for variant in formal_variants:
            assert variant.dimensions["formality"] == "formal"

    def test_get_dimensions(self):
        """Test getting all dimensions."""
        prompts = SentimentPrompts("test_task")

        dimensions = prompts.get_dimensions()

        expected_dimensions = {
            "formality": ["formal", "casual"],
            "phrasing": ["imperative", "question"],
            "order": ["task_first", "text_first"],
            "synonyms": ["set_a", "set_b"],
        }

        assert set(dimensions.keys()) == set(expected_dimensions.keys())

        for dim_name, values in expected_dimensions.items():
            assert set(dimensions[dim_name]) == set(values)


class TestPromptPerturbator:
    """Test the PromptPerturbator class."""

    def test_perturbator_initialization(self, sentiment_prompts):
        """Test PromptPerturbator initialization."""
        perturbator = PromptPerturbator(sentiment_prompts)

        assert perturbator.template == sentiment_prompts
        assert len(perturbator.dimensions) == 4

    def test_get_baseline_variants(self, prompt_perturbator):
        """Test getting baseline variants."""
        variants = prompt_perturbator.get_baseline_variants()

        assert len(variants) == 16
        assert all(hasattr(variant, "id") for variant in variants)

    def test_get_dimensional_comparison(self, prompt_perturbator):
        """Test getting dimensional comparison."""
        formality_groups = prompt_perturbator.get_dimensional_comparison("formality")

        assert "formal" in formality_groups
        assert "casual" in formality_groups
        assert len(formality_groups["formal"]) == 8
        assert len(formality_groups["casual"]) == 8

    def test_get_dimensional_comparison_invalid(self, prompt_perturbator):
        """Test error handling for invalid dimension."""
        with pytest.raises(ValueError, match="Dimension 'invalid' not found"):
            prompt_perturbator.get_dimensional_comparison("invalid")

    def test_get_formality_comparison(self, prompt_perturbator):
        """Test getting formality comparison."""
        comparison = prompt_perturbator.get_formality_comparison()

        assert "formal" in comparison
        assert "casual" in comparison
        assert len(comparison["formal"]) == 8
        assert len(comparison["casual"]) == 8

    def test_get_phrasing_comparison(self, prompt_perturbator):
        """Test getting phrasing comparison."""
        comparison = prompt_perturbator.get_phrasing_comparison()

        assert "imperative" in comparison
        assert "question" in comparison
        assert len(comparison["imperative"]) == 8
        assert len(comparison["question"]) == 8

    def test_get_order_comparison(self, prompt_perturbator):
        """Test getting order comparison."""
        comparison = prompt_perturbator.get_order_comparison()

        assert "task_first" in comparison
        assert "text_first" in comparison
        assert len(comparison["task_first"]) == 8
        assert len(comparison["text_first"]) == 8

    def test_get_synonym_comparison(self, prompt_perturbator):
        """Test getting synonym comparison."""
        comparison = prompt_perturbator.get_synonym_comparison()

        assert "set_a" in comparison
        assert "set_b" in comparison
        assert len(comparison["set_a"]) == 8
        assert len(comparison["set_b"]) == 8

    def test_get_minimal_pairs(self, prompt_perturbator):
        """Test getting minimal pairs."""
        formality_pairs = prompt_perturbator.get_minimal_pairs("formality")

        # Should have pairs that differ only in formality
        assert len(formality_pairs) > 0

        for variant1, variant2 in formality_pairs:
            # Check that they differ only in formality
            diff_count = 0
            for dim_name in variant1.dimensions:
                if variant1.dimensions[dim_name] != variant2.dimensions[dim_name]:
                    diff_count += 1

            assert diff_count == 1
            assert variant1.dimensions["formality"] != variant2.dimensions["formality"]

    def test_get_experimental_groups(self, prompt_perturbator):
        """Test getting experimental groups."""
        groups = prompt_perturbator.get_experimental_groups()

        expected_groups = [
            "all_baseline",
            "formal_only",
            "casual_only",
            "imperative_only",
            "question_only",
            "task_first_only",
            "text_first_only",
            "synonym_a_only",
            "synonym_b_only",
        ]

        assert set(groups.keys()) == set(expected_groups)
        assert len(groups["all_baseline"]) == 16
        assert len(groups["formal_only"]) == 8
        assert len(groups["casual_only"]) == 8

    def test_get_cross_dimensional_analysis(self, prompt_perturbator):
        """Test getting cross-dimensional analysis."""
        analysis = prompt_perturbator.get_cross_dimensional_analysis()

        assert "formality_x_phrasing" in analysis
        assert "order_x_synonyms" in analysis

        formality_phrasing = analysis["formality_x_phrasing"]
        expected_keys = [
            "formal_imperative",
            "formal_question",
            "casual_imperative",
            "casual_question",
        ]

        assert set(formality_phrasing.keys()) == set(expected_keys)

        # Each combination should have 4 variants (2x2 from other dimensions)
        for key, variants in formality_phrasing.items():
            assert len(variants) == 4

    def test_get_variant_by_dimensions(self, prompt_perturbator):
        """Test getting variant by dimensions."""
        variant = prompt_perturbator.get_variant_by_dimensions(
            formality="formal",
            phrasing="imperative",
            order="task_first",
            synonyms="set_a",
        )

        assert variant is not None
        assert variant.id == "v1"
        assert variant.dimensions["formality"] == "formal"
        assert variant.dimensions["phrasing"] == "imperative"

    def test_get_variant_by_dimensions_not_found(self, prompt_perturbator):
        """Test getting variant by dimensions when not found."""
        variant = prompt_perturbator.get_variant_by_dimensions(
            formality="invalid", phrasing="imperative"
        )

        assert variant is None

    def test_get_robustness_test_groups(self, prompt_perturbator):
        """Test getting robustness test groups."""
        groups = prompt_perturbator.get_robustness_test_groups()

        expected_groups = [
            "high_formality",
            "low_formality",
            "direct_commands",
            "questions",
            "context_first",
            "instruction_first",
            "technical_terms",
            "common_terms",
        ]

        assert set(groups.keys()) == set(expected_groups)
        assert len(groups["high_formality"]) == 8  # formal variants
        assert len(groups["low_formality"]) == 8  # casual variants

    def test_format_experimental_group(self, prompt_perturbator):
        """Test formatting experimental group."""
        formatted = prompt_perturbator.format_experimental_group(
            "formal_only", input_text="Test movie review"
        )

        assert len(formatted) == 8  # 8 formal variants
        assert all("Test movie review" in prompt for prompt in formatted.values())

    def test_format_experimental_group_invalid(self, prompt_perturbator):
        """Test error handling for invalid group."""
        with pytest.raises(ValueError, match="Group 'invalid_group' not found"):
            prompt_perturbator.format_experimental_group(
                "invalid_group", input_text="test"
            )
