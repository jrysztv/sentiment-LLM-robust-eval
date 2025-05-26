"""
Phase 2 Context Enhancement Module for Deep Learning Final Assignment.

This module implements the context enhancement system that:
1. Selects diverse few-shot examples from the training split
2. Creates prefix and suffix context-enhanced variants for best-performing models
3. Ensures no data contamination between train and validation splits
4. Provides comprehensive data persistence and audit trails

Key Features:
- Length and TF-IDF based diverse example selection
- Multi-model best variant selection (one per model)
- Prefix and suffix context integration
- Complete contamination validation
- Rich analytical data capture
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import SST5Loader, DataSample, SentimentLabels
from .prompts import SentimentPrompts
from .prompts.template import PromptTemplate, PromptVariant
from .evaluation import RobustnessEvaluator
from .evaluation.metrics import EvaluationResult


@dataclass
class ContextExample:
    """Represents a context example with metadata."""

    text: str
    label: str
    length_category: str  # "short", "medium", "long"
    tfidf_dissimilarity_score: float
    selection_reason: str
    source_split: str = "train"


@dataclass
class ContextSelectionAudit:
    """Audit trail for context selection process."""

    timestamp: str
    selection_strategy: str
    data_split_used: str
    total_train_samples: int
    selected_examples_per_label: int
    total_context_examples: int
    validation_split_used: str
    test_samples: int
    contamination_check: str  # "PASSED" or "FAILED"
    selected_context_examples: Dict[str, List[ContextExample]]
    contamination_validation: Dict[str, Any]
    tfidf_analysis: Dict[str, Any]


@dataclass
class EnhancedVariant:
    """Context-enhanced prompt variant."""

    variant_id: str  # e.g., "v17_prefix", "v18_suffix"
    template_structure: str  # description of template structure
    full_template: str  # complete template with context
    context_position: str  # "prefix" or "suffix"
    base_variant: str  # original variant ID this is based on
    context_examples_used: List[ContextExample]


class ContextSelector:
    """Handles diverse context example selection from training data."""

    def __init__(self, sst5_loader: SST5Loader):
        """
        Initialize context selector.

        Args:
            sst5_loader: SST5 data loader instance
        """
        self.sst5_loader = sst5_loader
        self.vectorizer = None
        self.train_samples = None

    def _get_length_category(self, text: str) -> str:
        """Categorize text length."""
        length = len(text.split())
        if length <= 8:
            return "short"
        elif length <= 20:
            return "medium"
        else:
            return "long"

    def _load_train_samples(self) -> List[DataSample]:
        """Load training samples if not already loaded."""
        if self.train_samples is None:
            self.train_samples = self.sst5_loader.load_split("train")
        return self.train_samples

    def select_diverse_examples(
        self, n_per_label: int = 3, random_seed: int = 42
    ) -> Dict[str, List[ContextExample]]:
        """
        Select diverse context examples using length + TF-IDF diversity.

        Args:
            n_per_label: Number of examples to select per sentiment label
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary mapping labels to selected context examples
        """
        np.random.seed(random_seed)

        # Load training data
        train_samples = self._load_train_samples()

        # Group by label
        samples_by_label = {}
        for sample in train_samples:
            if sample.label not in samples_by_label:
                samples_by_label[sample.label] = []
            samples_by_label[sample.label].append(sample)

        selected_examples = {}

        # TF-IDF analysis for diversity
        all_texts = [sample.text for sample in train_samples]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        for label in SentimentLabels.get_all_labels():
            if label not in samples_by_label:
                continue

            label_samples = samples_by_label[label]

            # Group by length category
            length_groups = {"short": [], "medium": [], "long": []}
            for sample in label_samples:
                length_cat = self._get_length_category(sample.text)
                length_groups[length_cat].append(sample)

            # Select one from each length category if possible
            selected_for_label = []

            for length_cat in ["short", "medium", "long"]:
                if len(selected_for_label) >= n_per_label:
                    break

                candidates = length_groups[length_cat]
                if not candidates:
                    continue

                if len(selected_for_label) == 0:
                    # First selection: random
                    selected_sample = np.random.choice(candidates)
                    selected_for_label.append(selected_sample)
                else:
                    # Subsequent selections: maximize TF-IDF diversity
                    best_sample = None
                    best_dissimilarity = -1

                    # Get TF-IDF vectors for already selected samples
                    selected_texts = [s.text for s in selected_for_label]
                    selected_vectors = self.vectorizer.transform(selected_texts)

                    for candidate in candidates:
                        candidate_vector = self.vectorizer.transform([candidate.text])

                        # Calculate minimum cosine similarity with selected samples
                        similarities = cosine_similarity(
                            candidate_vector, selected_vectors
                        )[0]
                        min_similarity = min(similarities)
                        dissimilarity = 1 - min_similarity

                        if dissimilarity > best_dissimilarity:
                            best_dissimilarity = dissimilarity
                            best_sample = candidate

                    if best_sample:
                        selected_for_label.append(best_sample)

            # Convert to ContextExample objects
            context_examples = []
            for i, sample in enumerate(selected_for_label):
                length_cat = self._get_length_category(sample.text)

                # Calculate dissimilarity score
                if i == 0:
                    dissimilarity_score = 1.0  # First selection
                    selection_reason = f"first_selection_{length_cat}_text"
                else:
                    selected_texts = [s.text for s in selected_for_label[:i]]
                    selected_vectors = self.vectorizer.transform(selected_texts)
                    sample_vector = self.vectorizer.transform([sample.text])
                    similarities = cosine_similarity(sample_vector, selected_vectors)[0]
                    dissimilarity_score = 1 - min(similarities)
                    selection_reason = f"high_dissimilarity_{length_cat}_text"

                context_example = ContextExample(
                    text=sample.text,
                    label=sample.label,
                    length_category=length_cat,
                    tfidf_dissimilarity_score=dissimilarity_score,
                    selection_reason=selection_reason,
                    source_split="train",
                )
                context_examples.append(context_example)

            selected_examples[label] = context_examples

        return selected_examples

    def validate_no_contamination(
        self,
        context_examples: Dict[str, List[ContextExample]],
        test_samples: List[DataSample],
    ) -> Dict[str, Any]:
        """
        Validate that no context examples overlap with test samples.

        Args:
            context_examples: Selected context examples
            test_samples: Test samples from validation split

        Returns:
            Validation results dictionary

        Raises:
            ValueError: If contamination detected
        """
        # Create hashes for all context examples
        context_hashes = set()
        for label_examples in context_examples.values():
            for example in label_examples:
                text_hash = hashlib.md5(
                    example.text.strip().lower().encode()
                ).hexdigest()
                context_hashes.add(text_hash)

        # Create hashes for all test samples
        test_hashes = set()
        for sample in test_samples:
            text_hash = hashlib.md5(sample.text.strip().lower().encode()).hexdigest()
            test_hashes.add(text_hash)

        # Check for overlap
        overlap = context_hashes.intersection(test_hashes)

        validation_result = {
            "overlap_check_method": "md5_hash_comparison",
            "overlapping_examples": len(overlap),
            "validation_status": "NO_CONTAMINATION_DETECTED"
            if not overlap
            else "CONTAMINATION_DETECTED",
            "test_sample_hashes": list(test_hashes)[:50],  # First 50 for audit
            "context_example_hashes": list(context_hashes),
            "overlapping_hashes": list(overlap) if overlap else [],
        }

        if overlap:
            raise ValueError(
                f"CONTAMINATION DETECTED: {len(overlap)} overlapping examples found between context examples (train split) and test samples (validation split)"
            )

        return validation_result


class ContextEnhancedVariantGenerator:
    """Generates context-enhanced prompt variants."""

    def __init__(self, sentiment_prompts: SentimentPrompts):
        """
        Initialize variant generator.

        Args:
            sentiment_prompts: SentimentPrompts instance
        """
        self.sentiment_prompts = sentiment_prompts

    def _format_context_examples(
        self, examples: Dict[str, List[ContextExample]]
    ) -> str:
        """Format context examples for inclusion in prompts."""
        formatted_lines = ["Here are examples of sentiment classification:", ""]

        for label in SentimentLabels.get_all_labels():
            if label in examples:
                for example in examples[label]:
                    formatted_lines.append(f"{label}: '{example.text}'")

        return "\n".join(formatted_lines)

    def generate_enhanced_variants(
        self, base_variant_id: str, context_examples: Dict[str, List[ContextExample]]
    ) -> Tuple[EnhancedVariant, EnhancedVariant]:
        """
        Generate prefix and suffix context-enhanced variants.

        Args:
            base_variant_id: ID of the base variant to enhance
            context_examples: Selected context examples

        Returns:
            Tuple of (prefix_variant, suffix_variant)
        """
        # Get base variant
        base_variant = self.sentiment_prompts.get_variant(base_variant_id)
        base_template = base_variant.template

        # Format context examples
        context_text = self._format_context_examples(context_examples)

        # Flatten context examples for metadata
        all_context_examples = []
        for label_examples in context_examples.values():
            all_context_examples.extend(label_examples)

        # Generate prefix variant
        prefix_template = f"{context_text}\n\n{base_template}"
        prefix_variant = EnhancedVariant(
            variant_id=f"{base_variant_id}_prefix",
            template_structure="[CONTEXT_EXAMPLES] + [ORIGINAL_PROMPT]",
            full_template=prefix_template,
            context_position="prefix",
            base_variant=base_variant_id,
            context_examples_used=all_context_examples,
        )

        # Generate suffix variant
        suffix_template = f"{base_template}\n\n{context_text}"
        suffix_variant = EnhancedVariant(
            variant_id=f"{base_variant_id}_suffix",
            template_structure="[ORIGINAL_PROMPT] + [CONTEXT_EXAMPLES]",
            full_template=suffix_template,
            context_position="suffix",
            base_variant=base_variant_id,
            context_examples_used=all_context_examples,
        )

        return prefix_variant, suffix_variant


class Phase2ContextEnhancement:
    """Main coordinator for Phase 2 context enhancement."""

    def __init__(self, sst5_loader: SST5Loader, sentiment_prompts: SentimentPrompts):
        """
        Initialize Phase 2 context enhancement system.

        Args:
            sst5_loader: SST5 data loader
            sentiment_prompts: Sentiment prompts system
        """
        self.sst5_loader = sst5_loader
        self.sentiment_prompts = sentiment_prompts
        self.context_selector = ContextSelector(sst5_loader)
        self.variant_generator = ContextEnhancedVariantGenerator(sentiment_prompts)
        self.evaluator = RobustnessEvaluator()

    def _get_serializable_tfidf_analysis(self) -> Dict[str, Any]:
        """Get serializable TF-IDF analysis data."""
        if not self.context_selector.vectorizer:
            return {
                "vectorizer_params": {},
                "vocabulary_size": 0,
                "selection_algorithm": "greedy_max_dissimilarity",
            }

        # Get params and filter out non-serializable ones
        params = self.context_selector.vectorizer.get_params()
        serializable_params = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_params[key] = value
            elif hasattr(value, "__name__"):  # Function or class
                serializable_params[key] = str(value.__name__)
            else:
                serializable_params[key] = str(value)

        return {
            "vectorizer_params": serializable_params,
            "vocabulary_size": len(self.context_selector.vectorizer.vocabulary_),
            "selection_algorithm": "greedy_max_dissimilarity",
        }

    def run_context_enhancement(
        self,
        baseline_results: Dict[str, Dict[str, EvaluationResult]],
        test_samples: List[DataSample],
        n_context_per_label: int = 3,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run complete Phase 2 context enhancement process.

        Args:
            baseline_results: Results from Phase 1 baseline experiment
            test_samples: Test samples (from validation split)
            n_context_per_label: Number of context examples per label
            random_seed: Random seed for reproducibility

        Returns:
            Complete Phase 2 results dictionary
        """
        # Step 1: Select best variants per model
        best_per_model = self.evaluator.get_best_combinations_per_model(
            baseline_results
        )

        # Step 2: Select diverse context examples
        context_examples = self.context_selector.select_diverse_examples(
            n_per_label=n_context_per_label, random_seed=random_seed
        )

        # Step 3: Validate no contamination
        contamination_validation = self.context_selector.validate_no_contamination(
            context_examples, test_samples
        )

        # Step 4: Create audit trail
        audit = ContextSelectionAudit(
            timestamp=datetime.now().isoformat(),
            selection_strategy="length_diversity_plus_tfidf",
            data_split_used="train",
            total_train_samples=len(self.context_selector._load_train_samples()),
            selected_examples_per_label=n_context_per_label,
            total_context_examples=sum(
                len(examples) for examples in context_examples.values()
            ),
            validation_split_used="validation",
            test_samples=len(test_samples),
            contamination_check=contamination_validation["validation_status"],
            selected_context_examples=context_examples,
            contamination_validation=contamination_validation,
            tfidf_analysis=self._get_serializable_tfidf_analysis(),
        )

        # Step 5: Generate enhanced variants for each best model
        enhanced_variants = {}
        for model_name, (best_variant_id, best_result) in best_per_model.items():
            prefix_variant, suffix_variant = (
                self.variant_generator.generate_enhanced_variants(
                    best_variant_id, context_examples
                )
            )

            enhanced_variants[model_name] = {
                "base_variant": {
                    "variant_id": best_variant_id,
                    "weighted_index": float(best_result.weighted_index),
                    "custom_accuracy": float(best_result.custom_accuracy),
                    "consistency_score": float(best_result.consistency_score),
                },
                "prefix_variant": prefix_variant,
                "suffix_variant": suffix_variant,
            }

        return {
            "audit": audit,
            "best_baseline_combinations": best_per_model,
            "enhanced_variants": enhanced_variants,
            "context_examples": context_examples,
            "validation_results": contamination_validation,
        }

    def save_phase2_results(
        self, phase2_results: Dict[str, Any], output_dir: str = "results"
    ) -> Dict[str, str]:
        """
        Save Phase 2 results to comprehensive output files.

        Args:
            phase2_results: Complete Phase 2 results
            output_dir: Output directory

        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # 1. Context selection audit trail
        audit_file = output_path / f"context_selection_audit_{timestamp}.json"
        audit_data = {
            "context_selection_metadata": {
                "timestamp": phase2_results["audit"].timestamp,
                "selection_strategy": phase2_results["audit"].selection_strategy,
                "data_split_used": phase2_results["audit"].data_split_used,
                "total_train_samples": phase2_results["audit"].total_train_samples,
                "selected_examples_per_label": phase2_results[
                    "audit"
                ].selected_examples_per_label,
                "total_context_examples": phase2_results[
                    "audit"
                ].total_context_examples,
                "validation_split_used": phase2_results["audit"].validation_split_used,
                "test_samples": phase2_results["audit"].test_samples,
                "contamination_check": phase2_results["audit"].contamination_check,
            },
            "selected_context_examples": {
                label: [
                    {
                        "text": ex.text,
                        "label": ex.label,
                        "length_category": ex.length_category,
                        "tfidf_dissimilarity_score": ex.tfidf_dissimilarity_score,
                        "selection_reason": ex.selection_reason,
                    }
                    for ex in examples
                ]
                for label, examples in phase2_results["context_examples"].items()
            },
            "contamination_validation": phase2_results["validation_results"],
            "tfidf_analysis": phase2_results["audit"].tfidf_analysis,
        }

        with open(audit_file, "w") as f:
            json.dump(audit_data, f, indent=2)
        saved_files["audit"] = str(audit_file)

        # 2. Enhanced prompt variants
        variants_file = output_path / f"context_enhanced_prompts_{timestamp}.json"
        variants_data = {
            "enhanced_variants_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(phase2_results["enhanced_variants"]),
                "variants_per_model": 2,  # prefix + suffix
            },
            "best_baseline_combinations": {
                model_name: {
                    "variant_id": best_variant_id,
                    "weighted_index": float(best_result.weighted_index),
                    "custom_accuracy": float(best_result.custom_accuracy),
                    "consistency_score": float(best_result.consistency_score),
                }
                for model_name, (best_variant_id, best_result) in phase2_results[
                    "best_baseline_combinations"
                ].items()
            },
            "context_enhanced_variants": {
                model_name: {
                    "base_variant": data["base_variant"],
                    "prefix_context": {
                        "variant_id": data["prefix_variant"].variant_id,
                        "template_structure": data["prefix_variant"].template_structure,
                        "full_template": data["prefix_variant"].full_template,
                        "context_position": data["prefix_variant"].context_position,
                        "base_variant": data["prefix_variant"].base_variant,
                    },
                    "suffix_context": {
                        "variant_id": data["suffix_variant"].variant_id,
                        "template_structure": data["suffix_variant"].template_structure,
                        "full_template": data["suffix_variant"].full_template,
                        "context_position": data["suffix_variant"].context_position,
                        "base_variant": data["suffix_variant"].base_variant,
                    },
                }
                for model_name, data in phase2_results["enhanced_variants"].items()
            },
        }

        with open(variants_file, "w") as f:
            json.dump(variants_data, f, indent=2)
        saved_files["variants"] = str(variants_file)

        return saved_files
