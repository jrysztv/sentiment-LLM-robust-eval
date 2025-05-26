# Script to generate all assets for the analysis.md report
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
import os

warnings.filterwarnings("ignore")
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# Create assets directory
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)

# Load and process data (same as analysis.py)
results_dir = Path("results")
baseline_path = results_dir / "baseline_async_results_detailed_20250526_015953.json"
context_path = results_dir / "context_enhanced_results_detailed_20250526_150746.json"

# Load context enhancement metadata
with open(results_dir / "context_enhanced_prompts_20250526_020002.json", "r") as f:
    context_enhanced = json.load(f)

with open(results_dir / "context_selection_audit_20250526_020002.json", "r") as f:
    context_audit = json.load(f)


# Helper functions (same as analysis.py)
def get_base_variant_id(variant_id: str) -> str:
    return variant_id.replace("_prefix", "").replace("_suffix", "")


def get_context_position(variant_id: str):
    if variant_id.endswith("_prefix"):
        return "prefix"
    if variant_id.endswith("_suffix"):
        return "suffix"
    return None


def get_variant_dimensions(base_variant_id: str) -> dict:
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
    if not variant_predictions:
        return 0.0
    prediction_counts = Counter(variant_predictions)
    max_count = max(prediction_counts.values())
    return max_count / len(variant_predictions)


def calculate_group_consistency(variant_predictions: Dict[str, List[str]]) -> float:
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


def json_to_dataframe(detailed_json: dict) -> list[dict]:
    rows = []
    for model_name, model_data in detailed_json.items():
        for variant_id, variant_data in model_data.items():
            base_variant_id = get_base_variant_id(variant_id)
            ctx_position = get_context_position(variant_id)
            variant_dims = get_variant_dimensions(base_variant_id)

            rows.append(
                {
                    "model": model_name,
                    "variant_id": variant_id,
                    "base_variant_id": base_variant_id,
                    "is_enhanced": ctx_position is not None,
                    "context_position": ctx_position,
                    "custom_accuracy": variant_data["custom_accuracy"],
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
                    **variant_dims,
                    "predictions": variant_data["predictions"],
                    "true_labels": variant_data["true_labels"],
                    "total_samples": variant_data["metadata"]["total_samples"],
                }
            )
    return rows


# Load and process data
with open(baseline_path) as f:
    baseline_json = json.load(f)
with open(context_path) as f:
    context_json = json.load(f)

baseline_rows = json_to_dataframe(baseline_json)
context_rows = json_to_dataframe(context_json)

results_df = pd.concat(
    [pd.DataFrame(baseline_rows), pd.DataFrame(context_rows)], ignore_index=True
)

baseline_df = results_df[~results_df["is_enhanced"]].copy()
enhanced_df = results_df[results_df["is_enhanced"]].copy()

# Calculate model consistency
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

# Calculate weighted index
baseline_df["weighted_index"] = (
    0.7 * baseline_df["custom_accuracy"] + 0.3 * baseline_df["model_consistency"]
)
enhanced_df["weighted_index"] = (
    0.7 * enhanced_df["custom_accuracy"] + 0.3 * enhanced_df["model_consistency"]
)

results_df = pd.concat([baseline_df, enhanced_df], ignore_index=True)

print("📊 Data processing complete. Generating visualizations and tables...")

# 1. Generate Model Performance Summary Table
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

# Save model summary table
with open(assets_dir / "model_summary.md", "w", encoding="utf-8") as f:
    f.write("## Model Performance Summary\n\n")
    f.write(model_summary.to_markdown())

# 2. Generate Best/Worst Combinations Tables
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

with open(assets_dir / "best_worst_combinations.md", "w", encoding="utf-8") as f:
    f.write("## Best Performing Combinations\n\n")
    f.write(best_combinations.to_markdown(index=False))
    f.write("\n\n## Worst Performing Combinations\n\n")
    f.write(worst_combinations.to_markdown(index=False))

# 3. Generate Accuracy Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, model in enumerate(baseline_df["model"].unique()):
    model_data = baseline_df[baseline_df["model"] == model]

    heatmap_data = model_data.pivot_table(
        values="custom_accuracy",
        index=["formality", "phrasing"],
        columns=["order", "synonyms"],
        aggfunc="mean",
    )

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
plt.savefig(assets_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. Generate Dimensional Analysis
def create_dimensional_analysis(baseline_df):
    dimensional_results = []
    for dimension in ["formality", "phrasing", "order", "synonyms"]:
        for model in baseline_df["model"].unique():
            model_data = baseline_df[baseline_df["model"] == model]
            values = sorted(model_data[dimension].unique())
            if len(values) == 2:
                value1, value2 = values
                subset1 = model_data[model_data[dimension] == value1]
                subset2 = model_data[model_data[dimension] == value2]

                acc1_mean, acc1_std = (
                    subset1["custom_accuracy"].mean(),
                    subset1["custom_accuracy"].std(),
                )
                acc2_mean, acc2_std = (
                    subset2["custom_accuracy"].mean(),
                    subset2["custom_accuracy"].std(),
                )

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
                    }
                )
    return pd.DataFrame(dimensional_results)


dimensional_analysis_df = create_dimensional_analysis(baseline_df)

# Save dimensional analysis table
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

dimensional_display = dimensional_analysis_df[display_cols].round(4)

with open(assets_dir / "dimensional_analysis.md", "w", encoding="utf-8") as f:
    f.write("## Dimensional Impact Analysis\n\n")
    f.write(dimensional_display.to_markdown(index=False))

# 5. Generate Robustness Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

dimensions = ["formality", "phrasing", "order", "synonyms"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

for i, dimension in enumerate(dimensions):
    ax = axes[i]

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
plt.savefig(assets_dir / "robustness_visualization.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. Generate Error Pattern Analysis
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
plt.savefig(assets_dir / "error_patterns.png", dpi=300, bbox_inches="tight")
plt.close()

# Save error summary table
error_summary = baseline_df.groupby("model")[error_cols].agg(["mean", "std"]).round(2)

with open(assets_dir / "error_summary.md", "w", encoding="utf-8") as f:
    f.write("## Error Pattern Summary\n\n")
    f.write(error_summary.to_markdown())

# 7. Generate Context Enhancement Analysis
enhancement_analysis = []

if not enhanced_df.empty:
    for model in enhanced_df["model"].unique():
        model_baseline = baseline_df[baseline_df["model"] == model]
        model_enhanced = enhanced_df[enhanced_df["model"] == model]

        best_baseline = model_baseline.loc[model_baseline["weighted_index"].idxmax()]

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

enhancement_df = pd.DataFrame(enhancement_analysis)

if not enhancement_df.empty:
    # Save enhancement table
    with open(assets_dir / "context_enhancement.md", "w", encoding="utf-8") as f:
        f.write("## Context Enhancement Analysis\n\n")
        f.write(enhancement_df.round(4).to_markdown(index=False))

    # Generate context enhancement visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    plt.savefig(assets_dir / "context_enhancement.png", dpi=300, bbox_inches="tight")
    plt.close()

# 8. Generate Key Insights Summary
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

with open(assets_dir / "key_insights.md", "w", encoding="utf-8") as f:
    f.write("## Key Research Insights\n\n")
    f.write(insights_series.to_markdown())

# 9. Generate Context Distribution Table
context_examples = context_audit["selected_context_examples"]
context_summary = {}
for label, examples in context_examples.items():
    lengths = [ex["length_category"] for ex in examples]
    context_summary[label] = Counter(lengths)

context_dist_df = pd.DataFrame(context_summary).fillna(0).astype(int)

with open(assets_dir / "context_distribution.md", "w", encoding="utf-8") as f:
    f.write("## Context Examples Distribution\n\n")
    f.write(context_dist_df.to_markdown())

print("✅ All assets generated successfully!")
print(f"   📊 Tables saved: {len(list(assets_dir.glob('*.md')))} markdown files")
print(f"   📈 Plots saved: {len(list(assets_dir.glob('*.png')))} image files")
print("   📝 Ready to generate analysis.md report")
