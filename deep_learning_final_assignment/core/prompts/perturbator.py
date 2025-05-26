"""
Prompt perturbation strategies and experimental grouping.
"""

from typing import Dict, List, Set, Optional, Tuple
from .template import PromptTemplate, PromptVariant
from .sentiment_prompts import SentimentPrompts


class PromptPerturbator:
    """Manages systematic prompt perturbation strategies and experimental groupings."""

    def __init__(self, template: PromptTemplate):
        self.template = template
        self.dimensions = template.get_dimensions()

    def get_baseline_variants(self) -> List[PromptVariant]:
        """Get all 16 baseline prompt variants."""
        return self.template.get_all_variants()

    def get_dimensional_comparison(
        self, dimension: str
    ) -> Dict[str, List[PromptVariant]]:
        """
        Get variants grouped by a specific dimension for comparison.

        Args:
            dimension: The dimension to group by (formality, phrasing, order, synonyms)

        Returns:
            Dictionary mapping dimension values to lists of variants
        """
        if dimension not in self.dimensions:
            raise ValueError(
                f"Dimension '{dimension}' not found. Available: {list(self.dimensions.keys())}"
            )

        groups = {}
        for value in self.dimensions[dimension]:
            groups[value] = self.template.get_variants_by_dimension(dimension, value)

        return groups

    def get_formality_comparison(self) -> Dict[str, List[PromptVariant]]:
        """Get variants grouped by formality (formal vs casual)."""
        return self.get_dimensional_comparison("formality")

    def get_phrasing_comparison(self) -> Dict[str, List[PromptVariant]]:
        """Get variants grouped by phrasing (imperative vs question)."""
        return self.get_dimensional_comparison("phrasing")

    def get_order_comparison(self) -> Dict[str, List[PromptVariant]]:
        """Get variants grouped by order (task_first vs text_first)."""
        return self.get_dimensional_comparison("order")

    def get_synonym_comparison(self) -> Dict[str, List[PromptVariant]]:
        """Get variants grouped by synonym set (set_a vs set_b)."""
        return self.get_dimensional_comparison("synonyms")

    def get_minimal_pairs(
        self, dimension: str
    ) -> List[Tuple[PromptVariant, PromptVariant]]:
        """
        Get minimal pairs that differ only in the specified dimension.

        Args:
            dimension: The dimension that should differ between pairs

        Returns:
            List of tuples containing variant pairs that differ only in the specified dimension
        """
        variants = self.template.get_all_variants()
        pairs = []

        for i, variant1 in enumerate(variants):
            for variant2 in variants[i + 1 :]:
                # Check if they differ only in the specified dimension
                diff_count = 0
                diff_dimension = None

                for dim_name in variant1.dimensions:
                    if variant1.dimensions[dim_name] != variant2.dimensions[dim_name]:
                        diff_count += 1
                        diff_dimension = dim_name

                # If they differ in exactly one dimension and it's the target dimension
                if diff_count == 1 and diff_dimension == dimension:
                    pairs.append((variant1, variant2))

        return pairs

    def get_experimental_groups(self) -> Dict[str, List[PromptVariant]]:
        """
        Get predefined experimental groups for systematic evaluation.

        Returns:
            Dictionary mapping group names to lists of variants
        """
        return {
            "all_baseline": self.get_baseline_variants(),
            "formal_only": self.template.get_variants_by_dimension(
                "formality", "formal"
            ),
            "casual_only": self.template.get_variants_by_dimension(
                "formality", "casual"
            ),
            "imperative_only": self.template.get_variants_by_dimension(
                "phrasing", "imperative"
            ),
            "question_only": self.template.get_variants_by_dimension(
                "phrasing", "question"
            ),
            "task_first_only": self.template.get_variants_by_dimension(
                "order", "task_first"
            ),
            "text_first_only": self.template.get_variants_by_dimension(
                "order", "text_first"
            ),
            "synonym_a_only": self.template.get_variants_by_dimension(
                "synonyms", "set_a"
            ),
            "synonym_b_only": self.template.get_variants_by_dimension(
                "synonyms", "set_b"
            ),
        }

    def get_cross_dimensional_analysis(
        self,
    ) -> Dict[str, Dict[str, List[PromptVariant]]]:
        """
        Get variants organized for cross-dimensional analysis.

        Returns:
            Nested dictionary for analyzing interactions between dimensions
        """
        analysis = {}

        # Formality × Phrasing
        analysis["formality_x_phrasing"] = {}
        for formality in ["formal", "casual"]:
            for phrasing in ["imperative", "question"]:
                key = f"{formality}_{phrasing}"
                variants = []
                for variant in self.template.get_all_variants():
                    if (
                        variant.dimensions["formality"] == formality
                        and variant.dimensions["phrasing"] == phrasing
                    ):
                        variants.append(variant)
                analysis["formality_x_phrasing"][key] = variants

        # Order × Synonyms
        analysis["order_x_synonyms"] = {}
        for order in ["task_first", "text_first"]:
            for synonyms in ["set_a", "set_b"]:
                key = f"{order}_{synonyms}"
                variants = []
                for variant in self.template.get_all_variants():
                    if (
                        variant.dimensions["order"] == order
                        and variant.dimensions["synonyms"] == synonyms
                    ):
                        variants.append(variant)
                analysis["order_x_synonyms"][key] = variants

        return analysis

    def get_variant_by_dimensions(self, **dimensions) -> Optional[PromptVariant]:
        """
        Get a specific variant by its dimension values.

        Args:
            **dimensions: Keyword arguments specifying dimension values

        Returns:
            The matching variant or None if not found
        """
        for variant in self.template.get_all_variants():
            if all(
                variant.dimensions.get(dim) == value
                for dim, value in dimensions.items()
            ):
                return variant
        return None

    def get_robustness_test_groups(self) -> Dict[str, List[PromptVariant]]:
        """
        Get specific groups for robustness testing.

        Returns:
            Dictionary with test groups designed for robustness evaluation
        """
        return {
            "high_formality": self.template.get_variants_by_dimension(
                "formality", "formal"
            ),
            "low_formality": self.template.get_variants_by_dimension(
                "formality", "casual"
            ),
            "direct_commands": self.template.get_variants_by_dimension(
                "phrasing", "imperative"
            ),
            "questions": self.template.get_variants_by_dimension(
                "phrasing", "question"
            ),
            "context_first": self.template.get_variants_by_dimension(
                "order", "text_first"
            ),
            "instruction_first": self.template.get_variants_by_dimension(
                "order", "task_first"
            ),
            "technical_terms": self.template.get_variants_by_dimension(
                "synonyms", "set_a"
            ),
            "common_terms": self.template.get_variants_by_dimension(
                "synonyms", "set_b"
            ),
        }

    def format_experimental_group(self, group_name: str, **kwargs) -> Dict[str, str]:
        """
        Format all variants in an experimental group with provided variables.

        Args:
            group_name: Name of the experimental group
            **kwargs: Variables to substitute in templates

        Returns:
            Dictionary mapping variant IDs to formatted prompts
        """
        groups = self.get_experimental_groups()
        if group_name not in groups:
            raise ValueError(
                f"Group '{group_name}' not found. Available: {list(groups.keys())}"
            )

        formatted = {}
        for variant in groups[group_name]:
            formatted[variant.id] = variant.format(**kwargs)

        return formatted
