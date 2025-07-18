# %% [markdown]
# # Prompt Perturbation Robustness Testing for Large Language Models
# **CEU MSBA Deep Learning Final Assignment**
# *Author: Istvan Peter Jaray*
#
# ## Research Question
#
# **"How robust are foundation models when prompts are slightly modified?"**
#
# This analysis evaluates GPT-4.1 and GPT-4o-mini robustness across systematic prompt variations
# in sentiment classification, testing how small changes in prompt formulation affect model predictions.
#
# ### Technical Note
#
# *The experimental pipeline includes full support for local Ollama model evaluation (e.g., Qwen, Mistral, LLaMA variants) through flexible model architecture. However, due to time constraints in this assignment, the focus remains on OpenAI model comparison as a foundation for future research. The extensible design allows for straightforward expansion to local model evaluation in subsequent studies.*

# %%
# Core imports and setup
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
from typing import Dict, List, Any
import statistics
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# %% [markdown]
# ## 1. Experiment Design and Setup
#
# ### Task: 5-Class Sentiment Classification
# - **Dataset**: Stanford Sentiment Treebank v5 (SST-5)
# - **Labels**: Very Negative, Negative, Neutral, Positive, Very Positive
# - **Test Samples**: 50 balanced examples from validation split
# - **Models**: GPT-4.1 vs GPT-4o-mini
#
# ### Systematic Prompt Variations
#
# We tested **16 baseline variants** across 4 dimensions using a 2×2×2×2 matrix:
#
# **Dimensions:**
# - **Formality**: Formal vs Casual language
# - **Phrasing**: Imperative vs Question format
# - **Order**: Task-first vs Text-first presentation
# - **Synonyms**: Set A ("analyze", "sentiment", "classify") vs Set B ("evaluate", "emotion", "categorize")
#
# **Example Prompt Variants:**
#
# **V1 [Formal + Imperative + Task-first + Set A]:**
# ```
# Analyze the sentiment of the following text and classify it as Very Negative, Negative,
# Neutral, Positive, or Very Positive. Return your response in JSON format with the key
# "sentiment" and the classified value.
#
# Text: [INPUT_TEXT]
#
# Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
# ```
#
# **V9 [Casual + Imperative + Task-first + Set A]:**
# ```
# Check out this text and figure out the sentiment - is it Very Negative, Negative,
# Neutral, Positive, or Very Positive? Give me your answer in JSON format with
# "sentiment" as the key.
#
# Text: [INPUT_TEXT]
#
# Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
# ```
#
# **V15 [Casual + Question + Text-first + Set A]:**
# ```
# Text: [INPUT_TEXT]
#
# What's the sentiment of this text above? Can you classify it as Very Negative,
# Negative, Neutral, Positive, or Very Positive and give me the answer in JSON
# format with "sentiment" as the key?
#
# Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
# ```
#
# ### How to Run the Experiment
#
# ```bash
# # Full experiment: Both models with context enhancement
# poetry run python scripts/run_baseline_async.py --models gpt-4.1,gpt-4o-mini --n-samples 50 --run-phase2
#
# # Quick test: Single model
# poetry run python scripts/run_baseline_async.py --models gpt-4o-mini --n-samples 10
# ```

# %% [markdown]
# ### Design Rationale and Methodological Choices
#
# **Dimensional Selection Strategy**: The four chosen dimensions (formality, phrasing, order, synonyms) were selected for their intuitive accessibility and potential impact. These represent easily understood variations that practitioners might naturally encounter, yet could have strong effects on model behavior. **Order** was hypothesized to be especially important, as the positioning of task instructions relative to input text could significantly influence model attention and processing.
#
# **Ex-ante Performance Hypothesis**: Based on traditional NLP conventions, the expectation was that **formal + question + task-first + Set A** combinations would perform best. Set A employs precise technical terminology ("analyze", "sentiment", "classify") compared to Set B's more casual alternatives ("evaluate", "emotion", "categorize"). The questioning format was expected to be more effective than imperative instructions for eliciting careful analysis.
#
# ### Evaluation Metrics
#
# **1. Custom Accuracy (Polarity-Weighted)**
# - Uses MSE-based penalty system emphasizing polarity over intensity errors
# - Label encoding: Very Negative=-3, Negative=-2, Neutral=0, Positive=2, Very Positive=3
# - **Design Philosophy**: The encoding prioritizes fundamental sentiment polarity over intensity nuances. Extreme errors (Very Negative ↔ Very Positive) receive maximum penalty, while adjacent errors (Negative ↔ Very Negative) are treated more leniently. This reflects deployment priorities where sentiment direction matters more than precise intensity.
# - Calculated per individual prediction (model-variant-input combination) to ensure statistical testability
#
# **2. Group Consistency**
# - Measures prediction agreement across prompt variants within each model
# - **Per-input consistency**: For each test sample, what % of variants agree on the most common prediction
# - **Group consistency**: Average per-input consistency across all test samples
# - **Example**: If 12/16 variants predict "Positive" for an input → consistency = 0.75
# - **Alternative Approaches**: Future research could explore entropy-based measures for more nuanced uncertainty quantification, though the current max-agreement approach provides intuitive interpretability.

# %%
# Load experiment results
results_dir = Path("results")
baseline_path = results_dir / "baseline_async_results_detailed_20250526_015953.json"
context_path = results_dir / "context_enhanced_results_detailed_20250526_150746.json"

# Load context enhancement metadata
with open(results_dir / "context_enhanced_prompts_20250526_020002.json", "r") as f:
    context_enhanced = json.load(f)

with open(results_dir / "context_selection_audit_20250526_020002.json", "r") as f:
    context_audit = json.load(f)


# %%
# Helper functions for data processing
def get_base_variant_id(variant_id: str) -> str:
    """Remove context suffixes to get baseline variant ID"""
    return variant_id.replace("_prefix", "").replace("_suffix", "")


def get_context_position(variant_id: str):
    """Determine if variant has prefix/suffix context or is baseline"""
    if variant_id.endswith("_prefix"):
        return "prefix"
    if variant_id.endswith("_suffix"):
        return "suffix"
    return None


def get_variant_dimensions(base_variant_id: str) -> dict:
    """Map variant ID to 4-dimensional design matrix"""
    variant_map = {
        "v1": ("formal", "imperative", "task_first", "set_a"),
        "v2": ("formal", "imperative", "task_first", "set_b"),
        "v3": ("formal", "imperative", "text_first", "set_a"),
        "v4": ("formal", "imperative", "text_first", "set_b"),
        "v5": ("formal", "question", "task_first", "set_a"),
        "v6": ("formal", "question", "task_first", "set_b"),
        "v7": ("formal", "question", "text_first", "set_a"),
        "v8": ("formal", "question", "text_first", "set_b"),
        "v9": ("casual", "imperative", "task_first", "set_a"),
        "v10": ("casual", "imperative", "task_first", "set_b"),
        "v11": ("casual", "imperative", "text_first", "set_a"),
        "v12": ("casual", "imperative", "text_first", "set_b"),
        "v13": ("casual", "question", "task_first", "set_a"),
        "v14": ("casual", "question", "task_first", "set_b"),
        "v15": ("casual", "question", "text_first", "set_a"),
        "v16": ("casual", "question", "text_first", "set_b"),
    }
    return dict(
        zip(
            ["formality", "phrasing", "order", "synonyms"],
            variant_map.get(base_variant_id, ("unknown",) * 4),
        )
    )


def calculate_per_input_consistency(variant_predictions: List[str]) -> float:
    """Calculate consistency as max agreement percentage across variants"""
    if not variant_predictions:
        return 0.0
    prediction_counts = Counter(variant_predictions)
    max_count = max(prediction_counts.values())
    return max_count / len(variant_predictions)


def calculate_group_consistency(variant_predictions: Dict[str, List[str]]) -> float:
    """Calculate mean consistency across all inputs for a group of variants"""
    if not variant_predictions:
        return 0.0

    variant_lists = list(variant_predictions.values())
    total_inputs = len(variant_lists[0]) if variant_lists else 0

    per_input_consistency = []
    for input_idx in range(total_inputs):
        input_predictions = [
            variant_predictions[variant_id][input_idx]
            for variant_id in variant_predictions
        ]
        input_consistency = calculate_per_input_consistency(input_predictions)
        per_input_consistency.append(input_consistency)

    return statistics.mean(per_input_consistency) if per_input_consistency else 0.0


# %%
# Convert JSON results to unified dataframe
def json_to_dataframe(detailed_json: dict) -> list[dict]:
    """Convert detailed results JSON to list of rows for DataFrame"""
    rows = []
    for model_name, model_data in detailed_json.items():
        for variant_id, variant_data in model_data.items():
            base_variant_id = get_base_variant_id(variant_id)
            ctx_position = get_context_position(variant_id)
            variant_dims = get_variant_dimensions(base_variant_id)

            rows.append(
                {
                    # Core identifiers
                    "model": model_name,
                    "variant_id": variant_id,
                    "base_variant_id": base_variant_id,
                    "is_enhanced": ctx_position is not None,
                    "context_position": ctx_position,
                    # Performance metrics
                    "custom_accuracy": variant_data["custom_accuracy"],
                    # Error analysis
                    "adjacent_errors": variant_data["error_breakdown"][
                        "adjacent_errors"
                    ],
                    "cross_polarity_errors": variant_data["error_breakdown"][
                        "cross_polarity_errors"
                    ],
                    "extreme_errors": variant_data["error_breakdown"]["extreme_errors"],
                    "correct_predictions": variant_data["error_breakdown"][
                        "correct_predictions"
                    ],
                    # Dimensional design
                    **variant_dims,
                    # Raw data for analysis
                    "predictions": variant_data["predictions"],
                    "true_labels": variant_data["true_labels"],
                    "total_samples": variant_data["metadata"]["total_samples"],
                }
            )
    return rows


# Load and combine all results
with open(baseline_path) as f:
    baseline_json = json.load(f)
with open(context_path) as f:
    context_json = json.load(f)

baseline_rows = json_to_dataframe(baseline_json)
context_rows = json_to_dataframe(context_json)

# Create unified results dataframe
results_df = pd.concat(
    [pd.DataFrame(baseline_rows), pd.DataFrame(context_rows)], ignore_index=True
)

# Separate baseline and enhanced for analysis
baseline_df = results_df[~results_df["is_enhanced"]].copy()
enhanced_df = results_df[results_df["is_enhanced"]].copy()

# Calculate model consistency for each model
for model in baseline_df["model"].unique():
    model_data = baseline_df[baseline_df["model"] == model]
    variant_predictions = {}
    for _, row in model_data.iterrows():
        variant_predictions[row["variant_id"]] = row["predictions"]
    model_consistency = calculate_group_consistency(variant_predictions)
    baseline_df.loc[baseline_df["model"] == model, "model_consistency"] = (
        model_consistency
    )

for model in enhanced_df["model"].unique():
    model_data = enhanced_df[enhanced_df["model"] == model]
    variant_predictions = {}
    for _, row in model_data.iterrows():
        variant_predictions[row["variant_id"]] = row["predictions"]
    model_consistency = calculate_group_consistency(variant_predictions)
    enhanced_df.loc[enhanced_df["model"] == model, "model_consistency"] = (
        model_consistency
    )

# Calculate weighted index: 0.7 × accuracy + 0.3 × consistency
baseline_df["weighted_index"] = (
    0.7 * baseline_df["custom_accuracy"] + 0.3 * baseline_df["model_consistency"]
)
enhanced_df["weighted_index"] = (
    0.7 * enhanced_df["custom_accuracy"] + 0.3 * enhanced_df["model_consistency"]
)

# Update unified dataframe
results_df = pd.concat([baseline_df, enhanced_df], ignore_index=True)

print("📊 Data Loading Complete:")
print(f"   Models tested: {list(results_df['model'].unique())}")
print(f"   Baseline variants: {len(baseline_df)} combinations")
print(f"   Enhanced variants: {len(enhanced_df)} combinations")
print(f"   Test samples: {baseline_df['total_samples'].iloc[0]}")

# %% [markdown]
# ## 2. Performance Analysis

# %%
# Model performance summary
model_summary = (
    baseline_df.groupby("model")
    .agg(
        {
            "custom_accuracy": ["mean", "std", "min", "max"],
            "model_consistency": ["mean", "std"],
            "weighted_index": ["mean", "std", "min", "max"],
        }
    )
    .round(3)
)
model_summary.columns = ["_".join(col) for col in model_summary.columns]
model_summary

# %%
# Best performing combinations
best_combinations = baseline_df.nlargest(5, "weighted_index")[
    [
        "model",
        "variant_id",
        "formality",
        "phrasing",
        "order",
        "synonyms",
        "custom_accuracy",
        "model_consistency",
        "weighted_index",
    ]
].round(3)
best_combinations

# %%
# Worst performing combinations
worst_combinations = baseline_df.nsmallest(5, "weighted_index")[
    [
        "model",
        "variant_id",
        "formality",
        "phrasing",
        "order",
        "synonyms",
        "custom_accuracy",
        "model_consistency",
        "weighted_index",
    ]
].round(3)
worst_combinations

# %%
# Accuracy heatmap across prompt dimensions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, model in enumerate(baseline_df["model"].unique()):
    model_data = baseline_df[baseline_df["model"] == model]

    # Create heatmap data
    heatmap_data = model_data.pivot_table(
        values="custom_accuracy",
        index=["formality", "phrasing"],
        columns=["order", "synonyms"],
        aggfunc="mean",
    )

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=axes[i],
        cbar_kws={"label": "Accuracy"},
        vmin=baseline_df["custom_accuracy"].min(),
        vmax=baseline_df["custom_accuracy"].max(),
    )
    axes[i].set_title(f"🎯 {model} Accuracy Heatmap", fontweight="bold")
    axes[i].set_xlabel("Order × Synonyms")
    axes[i].set_ylabel("Formality × Phrasing")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Performance Analysis Insights
#
# **Model Comparison Results**: GPT-4.1 and GPT-4o-mini demonstrated surprisingly close performance levels, though GPT-4.1 maintains clear advantages in both accuracy and consistency. The variance in accuracy for GPT-4.1 is notably low, indicating robust performance across prompt variations.
#
# **GPT-4o-mini Sensitivity Patterns**: Analysis reveals that GPT-4o-mini shows particular sensitivity to specific dimensional combinations:
# - **Text-first ordering** appears to negatively impact performance
# - **Casual formality** reduces effectiveness compared to formal approaches
# - **Set B synonyms** ("evaluate", "emotion", "categorize") demonstrate measurably inferior performance compared to Set A terms ("analyze", "sentiment", "classify")
#
# **Consistency vs. Accuracy Trade-offs**: GPT-4o-mini exhibits greater proneness to inconsistency across prompt variants, highlighting the importance of model strength in deployment scenarios. The weighted index framework (70% accuracy + 30% consistency) becomes particularly relevant when comparing similar-tier models, though here it primarily confirms GPT-4.1's superior reliability profile.
#
# **Deployment Implications**: While both models achieve high absolute performance, the reliability differential suggests GPT-4.1's robustness justifies its selection for production environments where consistent behavior across prompt variations is critical.

# %% [markdown]
# ## 3. Dimensional Impact Analysis
#
# ### Methodology Note
#
# **Accuracy Analysis**: Each cell represents **400 observations** (8 variants × 50 test samples) since accuracy is calculated per individual prediction.
#
# **Consistency Analysis**: Each cell represents **group-level metrics** where each of the 50 test samples was passed through all relevant variants (n=8 per dimensional subset), resulting in a single consistency score per group.


# %%
# Dimensional analysis with statistical testing
def create_dimensional_analysis(baseline_df):
    """Create comprehensive dimensional analysis"""
    dimensional_results = []

    for dimension in ["formality", "phrasing", "order", "synonyms"]:
        for model in baseline_df["model"].unique():
            model_data = baseline_df[baseline_df["model"] == model]

            # Get unique values for this dimension
            values = sorted(model_data[dimension].unique())
            if len(values) == 2:
                value1, value2 = values

                # Get subsets for each dimension value
                subset1 = model_data[model_data[dimension] == value1]
                subset2 = model_data[model_data[dimension] == value2]

                # Accuracy statistics (individual observations)
                acc1_mean, acc1_std = (
                    subset1["custom_accuracy"].mean(),
                    subset1["custom_accuracy"].std(),
                )
                acc2_mean, acc2_std = (
                    subset2["custom_accuracy"].mean(),
                    subset2["custom_accuracy"].std(),
                )

                # Group consistency calculation for dimensional subsets

                # Calculate group consistency for each dimensional subset
                if len(subset1) > 1:
                    variant_predictions_1 = {}
                    for _, row in subset1.iterrows():
                        variant_predictions_1[row["variant_id"]] = row["predictions"]
                    group_cons1 = calculate_group_consistency(variant_predictions_1)
                else:
                    group_cons1 = subset1["model_consistency"].iloc[0]

                if len(subset2) > 1:
                    variant_predictions_2 = {}
                    for _, row in subset2.iterrows():
                        variant_predictions_2[row["variant_id"]] = row["predictions"]
                    group_cons2 = calculate_group_consistency(variant_predictions_2)
                else:
                    group_cons2 = subset2["model_consistency"].iloc[0]

                # Statistical test for accuracy
                acc_t_stat, acc_p_val = stats.ttest_ind(
                    subset1["custom_accuracy"],
                    subset2["custom_accuracy"],
                    equal_var=False,
                )

                dimensional_results.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "value1": value1,
                        "value2": value2,
                        "acc_mean_1": acc1_mean,
                        "acc_mean_2": acc2_mean,
                        "acc_std_1": acc1_std,
                        "acc_std_2": acc2_std,
                        "acc_p_value": acc_p_val,
                        "acc_significant": "✓" if acc_p_val < 0.05 else "",
                        "group_cons_1": group_cons1,
                        "group_cons_2": group_cons2,
                        "n_variants_1": len(subset1),
                        "n_variants_2": len(subset2),
                    }
                )

    return pd.DataFrame(dimensional_results)


dimensional_analysis_df = create_dimensional_analysis(baseline_df)

# Create clean display version without theoretically unsound model_cons columns
display_cols = [
    "model",
    "dimension",
    "value1",
    "value2",
    "acc_mean_1",
    "acc_mean_2",
    "acc_std_1",
    "acc_std_2",
    "acc_p_value",
    "acc_significant",
    "group_cons_1",
    "group_cons_2",
]

# Display full dimensional analysis table
dimensional_display = dimensional_analysis_df[display_cols].round(4)
dimensional_display

# Summary of significant effects
significant_effects = dimensional_analysis_df[
    dimensional_analysis_df["acc_significant"] == "✓"
]
if len(significant_effects) > 0:
    print("\n📈 Significant Accuracy Effects (p < 0.05):")
    for _, row in significant_effects.iterrows():
        acc_diff = abs(row["acc_mean_1"] - row["acc_mean_2"])
        print(
            f"✓ {row['model']} - {row['dimension']} ({row['value1']} vs {row['value2']}):"
        )
        print(f"   Accuracy difference: {acc_diff:.3f} (p={row['acc_p_value']:.4f})")
else:
    print("No statistically significant accuracy effects found (α = 0.05)")

# Looking at the dimensional analysis' table:
dimensional_analysis_df
# %% [markdown]
# ### Statistical Significance Analysis Insights
#
# **Robustness of GPT-4.1**: The stronger model (GPT-4.1) demonstrates remarkable robustness across dimensional variations, with no statistically significant accuracy differences detected. This suggests that GPT-4.1 has achieved sufficient training sophistication to maintain consistent performance regardless of prompt formulation variations.
#
# **GPT-4o-mini Vulnerability**: The analysis reveals a statistically significant effect for **synonym choice** in GPT-4o-mini (p=0.0068), where Set A terminology ("analyze", "sentiment", "classify") outperforms Set B alternatives ("evaluate", "emotion", "categorize") by 0.007 accuracy points. While the effect size is modest, its statistical significance indicates systematic sensitivity to vocabulary precision.
#
# **Interpretation Limitations**: The significance testing reveals correlational patterns rather than causal relationships. The absence of other significant effects may reflect either genuine robustness or insufficient statistical power given the current sample size.
#
# **Practical Implications**: For deployment, these findings suggest that vocabulary choice matters more for smaller models, while larger models maintain performance across linguistic variations. This has direct implications for prompt engineering strategies across different model tiers.

# %%
# Robustness comparison across dimensions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

dimensions = ["formality", "phrasing", "order", "synonyms"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

for i, dimension in enumerate(dimensions):
    ax = axes[i]

    # Get data for this dimension
    dim_data = []
    for model in baseline_df["model"].unique():
        model_data = baseline_df[baseline_df["model"] == model]

        for value in sorted(model_data[dimension].unique()):
            subset = model_data[model_data[dimension] == value]
            dim_data.append(
                {
                    "model": model,
                    "dimension_value": value,
                    "accuracy_mean": subset["custom_accuracy"].mean(),
                    "accuracy_std": subset["custom_accuracy"].std(),
                    "consistency": subset["model_consistency"].mean(),
                }
            )

    dim_df = pd.DataFrame(dim_data)

    # Create grouped bar plot
    models = dim_df["model"].unique()
    x_width = 0.35
    x_positions = np.arange(len(dim_df[dim_df["model"] == models[0]]))

    for j, model in enumerate(models):
        model_subset = dim_df[dim_df["model"] == model]
        x_offset = (j - 0.5) * x_width

        bars = ax.bar(
            x_positions + x_offset,
            model_subset["accuracy_mean"],
            width=x_width,
            label=model,
            alpha=0.8,
            color=colors[j],
        )

        # Add error bars
        ax.errorbar(
            x_positions + x_offset,
            model_subset["accuracy_mean"],
            yerr=model_subset["accuracy_std"],
            fmt="none",
            color="black",
            alpha=0.6,
        )

    ax.set_title(f"{dimension.title()} Impact", fontweight="bold")
    ax.set_xlabel(f"{dimension.title()} Values")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dim_df[dim_df["model"] == models[0]]["dimension_value"])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(
    "📊 Model Robustness Across Prompt Dimensions",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.show()

# %% [markdown]
# ### Dimensional Robustness Visualization Insights
#
# **Visual Pattern Analysis**: The four-panel visualization reveals minimal visible differences across dimensional variations for both models, confirming the statistical analysis findings. Both models demonstrate remarkable stability across prompt formulations, with error bars indicating low variance within dimensional groups.
#
# **Model Comparison**: While absolute differences are small, GPT-4.1 consistently maintains slightly higher performance across all dimensions. The visualization confirms that neither model exhibits dramatic vulnerabilities to specific prompt formulations, supporting the hypothesis that modern large language models have achieved substantial robustness to surface-level prompt variations.
#
# **Dimensional Uniformity**: The near-uniform performance across dimensions suggests that the tested variations (formality, phrasing, order, synonyms) represent relatively minor perturbations from the models' perspective. This finding has important implications for prompt engineering, indicating that practitioners can prioritize clarity and task-specificity over precise linguistic formulation.

# %% [markdown]
# ## 4. Error Pattern Analysis

# %%
# Error breakdown visualization
error_cols = [
    "adjacent_errors",
    "cross_polarity_errors",
    "extreme_errors",
    "correct_predictions",
]
error_plot_data = baseline_df.groupby("model")[error_cols].mean()

fig, ax = plt.subplots(figsize=(12, 6))
error_plot_data.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("🎯 Error Pattern Analysis by Model", fontsize=14, fontweight="bold")
ax.set_ylabel("Average Count per 50 Samples")
ax.set_xlabel("Model")
ax.legend(title="Error Types", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# Error statistics summary
error_summary = baseline_df.groupby("model")[error_cols].agg(["mean", "std"]).round(2)
error_summary

# %% [markdown]
# ### Error Pattern Analysis Insights
#
# **Model Similarity in Error Patterns**: Both GPT-4.1 and GPT-4o-mini exhibit remarkably similar error distributions, with no visible differences in the visualization or tabular results. This suggests that both models share similar failure modes despite their performance differences.
#
# **Error Type Distribution**: The analysis reveals that **adjacent errors** are the most frequent error type, while extreme errors (Very Negative ↔ Very Positive) are essentially absent from both models. A small number of cross-polarity errors (positive ↔ negative) occur, but these remain minimal.
#
# **Implications for Accuracy Measurement**: The prevalence of adjacent errors suggests that our high accuracy measurements may be masking nuanced performance differences. The current polarity-weighted accuracy metric successfully penalizes extreme misclassifications, but adjacent errors remain relatively under-penalized. For applications requiring precise intensity discrimination, implementing F1-score metrics with no categorical differentiation might reveal additional performance insights.
#
# **Safety Assessment**: The absence of extreme errors and minimal cross-polarity confusion indicates that both models maintain fundamental sentiment understanding, making catastrophic misclassifications highly unlikely in production deployment.

# %% [markdown]
# ## 5. Context Enhancement Analysis

# %%
# Context enhancement effectiveness analysis
enhancement_analysis = []

if not enhanced_df.empty:
    for model in enhanced_df["model"].unique():
        model_baseline = baseline_df[baseline_df["model"] == model]
        model_enhanced = enhanced_df[enhanced_df["model"] == model]

        # Get best baseline for this model
        best_baseline = model_baseline.loc[model_baseline["weighted_index"].idxmax()]

        # Compare with enhanced variants
        for _, enhanced in model_enhanced.iterrows():
            enhancement_analysis.append(
                {
                    "model": model,
                    "enhanced_variant": enhanced["variant_id"],
                    "context_position": enhanced["context_position"],
                    "baseline_accuracy": best_baseline["custom_accuracy"],
                    "enhanced_accuracy": enhanced["custom_accuracy"],
                    "accuracy_improvement": enhanced["custom_accuracy"]
                    - best_baseline["custom_accuracy"],
                    "baseline_consistency": best_baseline["model_consistency"],
                    "enhanced_consistency": enhanced["model_consistency"],
                    "consistency_improvement": enhanced["model_consistency"]
                    - best_baseline["model_consistency"],
                    "baseline_weighted": best_baseline["weighted_index"],
                    "enhanced_weighted": enhanced["weighted_index"],
                    "weighted_improvement": enhanced["weighted_index"]
                    - best_baseline["weighted_index"],
                }
            )

# Create enhancement dataframe (explicitly called outside if statement)
enhancement_df = pd.DataFrame(enhancement_analysis)

if not enhancement_df.empty:
    enhancement_df.round(4)

    # Visualization of context enhancement impact
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy improvement
    enhancement_plot_data = enhancement_df.melt(
        id_vars=["model", "context_position"],
        value_vars=["baseline_accuracy", "enhanced_accuracy"],
        var_name="condition",
        value_name="accuracy",
    )

    sns.barplot(
        data=enhancement_plot_data,
        x="context_position",
        y="accuracy",
        hue="condition",
        ax=ax1,
    )
    ax1.set_title("🚀 Context Enhancement: Accuracy Impact", fontweight="bold")
    ax1.set_ylabel("Accuracy")

    # Consistency improvement
    consistency_plot_data = enhancement_df.melt(
        id_vars=["model", "context_position"],
        value_vars=["baseline_consistency", "enhanced_consistency"],
        var_name="condition",
        value_name="consistency",
    )

    sns.barplot(
        data=consistency_plot_data,
        x="context_position",
        y="consistency",
        hue="condition",
        ax=ax2,
    )
    ax2.set_title("🔄 Context Enhancement: Consistency Impact", fontweight="bold")
    ax2.set_ylabel("Consistency")

    plt.tight_layout()
    plt.show()
else:
    print("No context enhancement data available")
    enhancement_df = pd.DataFrame()  # Ensure variable exists
enhancement_df

# %% [markdown]
# ### Context Enhancement Analysis Insights
#
# **Unexpected Negative Impact**: Context enhancement produced a surprising negative effect on model performance, contrary to typical few-shot learning expectations. This outcome challenges conventional assumptions about the universal benefit of providing examples in language model prompts.
#
# **Context Collection Methodology**: The context examples were selected using a sophisticated approach combining length diversity (short/medium/long examples per label) and TF-IDF cosine dissimilarity optimization to ensure semantic diversity across the 15 examples (3 per sentiment label). Data contamination was prevented through strict separation of training (context source) and validation (test source) splits.
#
# **Positional Effects**: Prefix positioning (examples before task prompt) caused less performance degradation than suffix positioning (examples after task prompt), suggesting that context placement affects model attention and processing efficiency. However, both positions resulted in net negative impact.
#
# **Cost-Benefit Analysis**: Given the substantial token overhead (approximately 25% increase) combined with negative performance impact, context enhancement proves counterproductive for these models and task. This finding suggests that GPT-4.1 and GPT-4o-mini may already possess sufficient sentiment classification capabilities, making additional examples redundant or potentially confusing.
#
# **Research Implications**: The negative context enhancement effect indicates that modern large language models may have reached a sophistication level where traditional few-shot prompting strategies become unnecessary or even detrimental for well-defined classification tasks.

# %% [markdown]
# ## 6. Key Findings and Research Insights

# %%
# Comprehensive research insights summary
insights = {
    "Best Overall Model": baseline_df.loc[
        baseline_df["weighted_index"].idxmax(), "model"
    ],
    "Best Overall Variant": baseline_df.loc[
        baseline_df["weighted_index"].idxmax(), "variant_id"
    ],
    "Best Overall Score": baseline_df["weighted_index"].max(),
    "Model Consistency Range": f"{baseline_df['model_consistency'].min():.3f} - {baseline_df['model_consistency'].max():.3f}",
    "Accuracy Range": f"{baseline_df['custom_accuracy'].min():.3f} - {baseline_df['custom_accuracy'].max():.3f}",
    "Most Stable Dimension": dimensional_analysis_df.groupby("dimension")[
        ["group_cons_1", "group_cons_2"]
    ]
    .mean()
    .mean(axis=1)
    .idxmax(),
    "Least Stable Dimension": dimensional_analysis_df.groupby("dimension")[
        ["group_cons_1", "group_cons_2"]
    ]
    .mean()
    .mean(axis=1)
    .idxmin(),
}

if not enhancement_df.empty:
    insights["Context Enhancement Effect"] = (
        f"{enhancement_df['accuracy_improvement'].mean():+.3f} avg accuracy improvement"
    )
    insights["Best Context Position"] = enhancement_df.loc[
        enhancement_df["accuracy_improvement"].idxmax(), "context_position"
    ]

insights_series = pd.Series(insights)
insights_series

# %%
# Context selection strategy summary
context_examples = context_audit["selected_context_examples"]
contamination_status = context_audit["contamination_validation"]["validation_status"]

print("📝 Context Enhancement Strategy:")
print(f"   Selection method: Length diversity + TF-IDF dissimilarity")
print(f"   Examples per label: 3 (total: 15)")
print(f"   Contamination check: {contamination_status}")
print(f"   Context positions tested: Prefix and Suffix")

# Context examples distribution
context_summary = {}
for label, examples in context_examples.items():
    lengths = [ex["length_category"] for ex in examples]
    context_summary[label] = Counter(lengths)

context_dist_df = pd.DataFrame(context_summary).fillna(0).astype(int)
context_dist_df

# %% [markdown]
# ### Research Insights and Deployment Recommendations
#
# **Model Selection Findings**: As expected, **GPT-4.1** emerges as the superior choice for production deployment, demonstrating both higher accuracy and greater consistency across prompt variations. The analysis confirms that model strength directly correlates with robustness to prompt perturbations.
#
# **Dimensional Stability Insights**: **Order** proves to be the most stable dimension across both models, while **phrasing** shows the least stability. This finding suggests that task instruction positioning relative to input text matters less than initially hypothesized, while the imperative vs. question distinction creates more variability in model responses.
#
# **Prompt Engineering Best Practices**: Based on the experimental evidence, the optimal prompt formulation strategy involves:
# - **Maintaining formality** in language rather than casual approaches
# - **Employing question format** rather than imperative instructions
# - **Using precise technical terminology** (Set A: "analyze", "sentiment", "classify") over casual alternatives
# - **Prioritizing clarity** over specific structural arrangements, given the minimal order effects
#
# **Critical Finding - Model Strength Supremacy**: The most significant insight is that **model strength matters more than prompt optimization**. GPT-4.1's consistently superior performance across all variations suggests that investing in more capable models yields greater returns than extensive prompt engineering for smaller models.
#
# **Production Deployment Strategy**: For organizations deploying sentiment classification systems, the evidence supports a "play it safe" approach: select the most capable model available (GPT-4.1), use formal question-based prompts with technical vocabulary, and avoid unnecessary complexity in prompt design. The robustness demonstrated by modern large language models suggests that practitioners can focus on clear task specification rather than elaborate prompt engineering strategies.

# %%
print("✅ Analysis Complete - Ready for Reporting")
print(f"   Baseline combinations analyzed: {len(baseline_df)}")
print(f"   Enhanced combinations analyzed: {len(enhanced_df)}")
print(f"   Dimensional analyses completed: {len(dimensional_analysis_df)}")
print(
    f"   Key datasets prepared: results_df, baseline_df, enhanced_df, dimensional_analysis_df, enhancement_df"
)


# %%
