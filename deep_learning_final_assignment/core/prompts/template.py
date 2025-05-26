"""
Base prompt template management system.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PromptVariant:
    """Represents a single prompt variant with metadata."""

    id: str
    name: str
    template: str
    dimensions: Dict[str, str]
    description: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)


class PromptTemplate(ABC):
    """Abstract base class for prompt template management."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.variants: Dict[str, PromptVariant] = {}
        self._setup_variants()

    @abstractmethod
    def _setup_variants(self) -> None:
        """Setup all prompt variants for this template."""
        pass

    def add_variant(self, variant: PromptVariant) -> None:
        """Add a prompt variant to the template."""
        self.variants[variant.id] = variant

    def get_variant(self, variant_id: str) -> PromptVariant:
        """Get a specific prompt variant by ID."""
        if variant_id not in self.variants:
            raise ValueError(f"Variant '{variant_id}' not found")
        return self.variants[variant_id]

    def get_variants_by_dimension(
        self, dimension: str, value: str
    ) -> List[PromptVariant]:
        """Get all variants that match a specific dimension value."""
        return [
            variant
            for variant in self.variants.values()
            if variant.dimensions.get(dimension) == value
        ]

    def get_all_variants(self) -> List[PromptVariant]:
        """Get all available prompt variants."""
        return list(self.variants.values())

    def get_variant_ids(self) -> List[str]:
        """Get all variant IDs."""
        return list(self.variants.keys())

    def format_variant(self, variant_id: str, **kwargs) -> str:
        """Format a specific variant with provided variables."""
        variant = self.get_variant(variant_id)
        return variant.format(**kwargs)

    def format_all_variants(self, **kwargs) -> Dict[str, str]:
        """Format all variants with provided variables."""
        return {
            variant_id: variant.format(**kwargs)
            for variant_id, variant in self.variants.items()
        }

    def get_dimensions(self) -> Dict[str, List[str]]:
        """Get all available dimensions and their possible values."""
        dimensions = {}
        for variant in self.variants.values():
            for dim_name, dim_value in variant.dimensions.items():
                if dim_name not in dimensions:
                    dimensions[dim_name] = []
                if dim_value not in dimensions[dim_name]:
                    dimensions[dim_name].append(dim_value)
        return dimensions

    def get_variant_count(self) -> int:
        """Get the total number of variants."""
        return len(self.variants)

    def __len__(self) -> int:
        """Return the number of variants."""
        return len(self.variants)

    def __iter__(self):
        """Iterate over all variants."""
        return iter(self.variants.values())

    def __contains__(self, variant_id: str) -> bool:
        """Check if a variant ID exists."""
        return variant_id in self.variants
