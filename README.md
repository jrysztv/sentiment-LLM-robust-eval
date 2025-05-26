# Prompt Perturbation Robustness Testing for Large Language Models

**CEU MSBA Deep Learning Final Assignment**  
*Author: Istvan Peter Jaray*

A comprehensive research framework for evaluating how foundation models respond to systematic variations in prompt phrasing. This project implements robustness testing through prompt perturbation analysis, comparing model consistency and accuracy across 16 different prompt formulations.

## 🎯 Research Question

**"How robust are foundation models when prompts are slightly modified?"**

We test this by systematically varying prompts across 4 dimensions:
- **Formality**: Formal vs Casual language
- **Phrasing**: Imperative vs Question format  
- **Order**: Task-first vs Text-first presentation
- **Synonyms**: Technical vs Common terminology

## 🏆 Key Research Findings

### **Model Performance Results**
- **GPT-4.1**: Superior consistency (95.5%) and accuracy (96.8%) across all prompt variations
- **GPT-4o-mini**: Good performance (89.6% consistency, 95.2% accuracy) but with notable sensitivity to synonym choice
- **Statistical Significance**: Only GPT-4o-mini showed significant sensitivity to vocabulary choice (p=0.0068)

### **Critical Insights**
1. **Model Strength > Prompt Engineering**: GPT-4.1's robustness across variations suggests investing in capable models yields better returns than extensive prompt optimization
2. **Context Enhancement Counterproductive**: Few-shot examples decreased performance, contrary to conventional expectations
3. **Dimensional Stability**: "Order" was most stable dimension; "Phrasing" showed least stability
4. **Error Patterns**: Both models showed similar failure modes, primarily adjacent classification errors

## 📊 Complete Analysis & Reports

This project includes comprehensive analysis and reporting:

### **Interactive Analysis**
- **`analysis.ipynb`**: Complete Jupyter notebook with 29 cells (14 markdown, 15 code)
- **Executable analysis**: Statistical testing, visualizations, and insights
- **Interactive exploration**: Modify parameters and re-run analysis sections

### **Publication-Ready Reports**
- **Academic Report**: Comprehensive findings with embedded tables and visualizations
- **4 High-Quality Visualizations**: Accuracy heatmaps, robustness patterns, error analysis, context enhancement impact
- **9 Data Tables**: Model summaries, dimensional analysis, best/worst combinations

### **Generated Assets**
```
assets/
├── accuracy_heatmap.png              # Model performance across dimensions
├── robustness_visualization.png      # Dimensional stability analysis  
├── error_patterns.png                # Error type distribution
├── context_enhancement.png           # Context impact analysis
├── model_summary.md                  # Performance statistics
├── dimensional_analysis.md           # Statistical significance testing
├── best_worst_combinations.md        # Top/bottom performing variants
├── context_enhancement.md            # Enhancement analysis results
└── key_insights.md                   # Research summary
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- OpenAI API key (for GPT models)

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/jrysztv/sentiment-LLM-robust-eval.git
cd sentiment-LLM-robust-eval
poetry install
```

2. **Configure API access**:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

3. **Run the complete experiment**:
```bash
# Primary research comparison: GPT-4.1 vs GPT-4o-mini with context enhancement
poetry run python scripts/run_baseline_async.py --models gpt-4.1,gpt-4o-mini --n-samples 50 --run-phase2

# Quick test run (smaller sample)
poetry run python scripts/run_baseline_async.py --models gpt-4o-mini --n-samples 10
```

4. **Run analysis and generate reports**:
```bash
# Execute interactive analysis
jupyter lab analysis.ipynb

# Generate all visualizations and tables
python generate_report_assets.py
```

## 📈 Analysis & Visualizations

### **Comprehensive Jupyter Notebook Analysis**
The `analysis.ipynb` notebook provides complete interactive analysis:

1. **Experimental Design**: Methodology, rationale, and evaluation metrics
2. **Performance Analysis**: Model comparison with statistical testing
3. **Dimensional Impact**: 4-dimensional robustness evaluation with significance testing
4. **Error Pattern Analysis**: Failure mode characterization
5. **Context Enhancement**: Few-shot prompting effectiveness evaluation
6. **Research Insights**: Deployment recommendations and best practices

### **Key Visualizations**
- **Accuracy Heatmaps**: Performance across formality×phrasing vs order×synonyms
- **Robustness Charts**: Dimensional stability comparison between models
- **Error Distribution**: Adjacent vs cross-polarity vs extreme error patterns  
- **Context Impact**: Prefix vs suffix enhancement effectiveness

### **Statistical Analysis**
- **Dimensional Testing**: T-tests across 4 prompt dimensions
- **Consistency Metrics**: Group agreement calculation across variants
- **Significance Detection**: p<0.05 effects identification
- **Effect Size Quantification**: Practical significance assessment

## 💡 Prompt Variation Strategy

### **Systematic 16-Variant Design**
Our systematic approach creates **16 baseline variants** using a 2×2×2×2 matrix:

#### **Dimension Examples**

**Formality Dimension**:
- **Formal**: *"Analyze the sentiment of the following text and classify it as..."*
- **Casual**: *"Check out this text and figure out the sentiment - is it..."*

**Phrasing Dimension**:
- **Imperative**: *"Analyze the sentiment..." / "Check out this text..."*
- **Question**: *"What is the sentiment...?" / "What's the sentiment...?"*

**Order Dimension**:
- **Task-first**: *"Analyze the sentiment... Text: [INPUT]"*
- **Text-first**: *"Text: [INPUT] Analyze the sentiment of the above text..."*

**Synonyms Dimension**:
- **Set A**: *"analyze", "sentiment", "classify"*
- **Set B**: *"evaluate", "emotion", "categorize"*

#### **Context Enhancement (Phase 2)**

After baseline testing, we enhance the **best-performing variant per model**:

1. **Best Variant Selection**: Identify highest weighted-index prompt for each model
2. **Context Addition**: Add 3 diverse examples per sentiment label (15 total examples)
3. **Positioning Assessment**: Test both prefix and suffix positioning

**Result**: **18 total variants** = 16 baseline + 2 context-enhanced

## 📊 Experimental Framework

### **Dataset: Stanford Sentiment Treebank v5 (SST-5)**
- **Source**: HuggingFace `SetFit/sst5` dataset
- **Labels**: 5-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)
- **Split Usage**:
  - `train` split: Context examples for Phase 2 enhancement (15 examples total)
  - `validation` split: Test samples for evaluation (50 balanced samples)
  - `test` split: Reserved (not used in current design)
- **Data Separation**: Zero contamination between context examples and test samples

### **Evaluation Metrics**
1. **Custom Polarity-Weighted Accuracy**: MSE-based penalty emphasizing cross-polarity errors
2. **Group Consistency**: Prediction agreement percentage across prompt variants
3. **Weighted Index**: 70% accuracy + 30% consistency for deployment ranking

### Phase 1: Baseline Robustness Testing
1. **Data Loading**: SST-5 sentiment dataset (balanced sampling from validation split)
2. **Prompt Generation**: 16 systematic variants across 4 dimensions
3. **Model Evaluation**: Tests all model-prompt combinations asynchronously
4. **Consistency Calculation**: Measures prediction agreement across variants

### Phase 2: Context Enhancement
1. **Best Variant Selection**: Identifies highest-performing prompt per model
2. **Context Selection**: Chooses diverse examples using length + TF-IDF diversity
3. **Enhanced Prompts**: Creates prefix/suffix context-enriched variants
4. **Order Assessment**: Determines optimal prompt element ordering

## 📁 Results & Output

### Experiment Results Location
All experiment outputs are saved in the `results/` directory with timestamp-based filenames:

```
results/
├── baseline_async_results_detailed_YYYYMMDD_HHMMSS.json    # Complete model predictions and metadata
├── baseline_async_results_summary_YYYYMMDD_HHMMSS.json     # Performance summaries by model
├── context_enhanced_results_detailed_YYYYMMDD_HHMMSS.json  # Context enhancement results
├── context_selection_audit_YYYYMMDD_HHMMSS.json           # Context selection process
└── context_enhanced_prompts_YYYYMMDD_HHMMSS.json          # Enhanced prompt variants
```

### Analysis Assets
```
assets/
├── *.png                              # 4 publication-quality visualizations
└── *.md                               # 9 formatted data tables
```

## 🧪 Testing

Comprehensive test suite with 208+ tests covering all functionality:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=deep_learning_final_assignment

# Fast tests only
poetry run pytest -m "not slow"
```

## ⚙️ Configuration

### Model Selection
```bash
# Compare specific models
--models gpt-4.1,gpt-4o-mini

# Single model testing  
--models gpt-4o-mini

# Local model support (Ollama) - Framework ready, not implemented in current study
--models qwen3:4b
```

### Experiment Parameters
- `--n-samples`: Number of test samples (default: 50)
- `--run-phase2`: Include context enhancement analysis
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## 🎯 Deployment Recommendations

Based on comprehensive analysis results:

### **Model Selection**
- **Choose GPT-4.1** for production environments requiring consistent performance
- **Model capability matters more than prompt optimization**

### **Prompt Engineering Best Practices**
- **Use formal language** rather than casual approaches
- **Employ question format** rather than imperative instructions
- **Use precise technical terminology** (Set A over Set B)
- **Prioritize clarity** over elaborate optimization

### **Context Strategy**
- **Avoid few-shot examples** for well-defined classification tasks with modern models
- **Context enhancement decreases performance** contrary to conventional expectations

## 📦 Project Structure

```
sentiment-LLM-robust-eval/
├── analysis.ipynb                     # 🔬 Complete interactive analysis
├── generate_report_assets.py          # 📊 Visualization and table generation
├── assets/                            # 📈 Generated visualizations and tables
├── scripts/
│   └── run_baseline_async.py          # 🚀 Main experiment entry point
├── deep_learning_final_assignment/    # Core framework
│   ├── core/
│   │   ├── models/                    # Model interfaces (OpenAI, Ollama)
│   │   ├── prompts/                   # 16-variant prompt system
│   │   ├── evaluation/                # Metrics and consistency calculation
│   │   ├── context_enhancement/       # Phase 2 context system
│   │   └── utils/                     # Logging and utilities
│   └── ...
├── results/                           # 📋 Experiment outputs
└── tests/                             # 🧪 Comprehensive test suite
```

## 🔬 Research Extensions

### **Technical Capabilities**
- **Local Model Support**: Framework includes Ollama integration for Qwen, Mistral, LLaMA evaluation
- **Extensible Design**: Easy expansion to additional tasks beyond sentiment classification
- **Scalable Pipeline**: Asynchronous processing supports large-scale experiments

### **Future Research Directions**
1. **Local Model Evaluation**: Expand to include Ollama-based models
2. **Cross-Task Validation**: Test robustness across different NLP tasks  
3. **Advanced Context Strategies**: Explore alternative few-shot selection methods
4. **Large-Scale Validation**: Increase sample sizes for higher statistical power

---

## 📜 Citation

If you use this research framework or findings, please cite:

```bibtex
@misc{jaray2025prompt,
  title={Prompt Perturbation Robustness Testing for Large Language Models},
  author={Jaray, Istvan Peter},
  year={2025},
  institution={Central European University, MSBA Program},
  type={Deep Learning Final Assignment}
}
```

---

*This comprehensive analysis demonstrates that modern large language models exhibit remarkable robustness to prompt variations, with model capability being more important than prompt engineering sophistication. The research provides practical insights for deployment strategies and challenges conventional assumptions about few-shot prompting effectiveness.*