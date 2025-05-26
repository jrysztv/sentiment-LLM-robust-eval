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

### **Prompt Variation Strategy**

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
3. **Ordering Assessment**: Test both prefix and suffix positioning

**Result**: **18 total variants** = 16 baseline + 2 context-enhanced (prefix + suffix)

### **Example Context Enhancement**
```
Best baseline variant: "Check out this text above and figure out the sentiment..."

→ Prefix variant: [15 examples] + [original prompt]
→ Suffix variant: [original prompt] + [15 examples]
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
poetry run python scripts/run_experiment.py --models gpt-4.1,gpt-4o-mini --n-samples 50 --run-phase2

# Quick test run (smaller sample)
poetry run python scripts/run_experiment.py --models gpt-4o-mini --n-samples 10 --run-phase2
```

## 📊 Experiment Pipeline

### **Dataset: Stanford Sentiment Treebank v5 (SST-5)**
- **Source**: HuggingFace `SetFit/sst5` dataset
- **Labels**: 5-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)
- **Split Usage**:
  - `train` split: Context examples for Phase 2 enhancement (15 examples total)
  - `validation` split: Test samples for evaluation (50 balanced samples)
  - `test` split: Reserved (not used in current design)
- **Data Separation**: Zero contamination between context examples and test samples

### Phase 1: Baseline Robustness Testing
1. **Data Loading**: SST-5 sentiment dataset (balanced sampling from validation split)
2. **Prompt Generation**: 16 systematic variants across 4 dimensions
3. **Model Evaluation**: Tests all model-prompt combinations asynchronously
4. **Consistency Calculation**: Measures prediction agreement across variants

### Phase 2: Context Enhancement (Optional)
1. **Best Variant Selection**: Identifies highest-performing prompt per model
2. **Context Selection**: Chooses diverse examples using length + TF-IDF diversity
3. **Enhanced Prompts**: Creates prefix/suffix context-enriched variants
4. **Order Assessment**: Determines optimal prompt element ordering
5. **Total Evaluation**: **18 variants** (16 baseline + 2 context-enhanced)

### Data Pipeline
```
SST-5 Dataset → Balanced Sampling → 16 Prompt Variants → Model Evaluation → Results Storage
     ↓                 ↓                    ↓                    ↓              ↓
Train/Validation   50 samples       GPT-4.1 + GPT-4o-mini   Async Batch    JSON Files
    splits        (balanced)         (32 combinations)      Processing     (Timestamped)
```

## 📁 Results & Output

### Experiment Results Location
All experiment outputs are saved in the `results/` directory with timestamp-based filenames:

```
results/
├── baseline_async_results_detailed_YYYYMMDD_HHMMSS.json    # Complete model predictions and metadata
├── baseline_async_results_summary_YYYYMMDD_HHMMSS.json     # Performance summaries by model
├── context_selection_audit_YYYYMMDD_HHMMSS.json           # Phase 2: Context selection process
└── context_enhanced_prompts_YYYYMMDD_HHMMSS.json          # Phase 2: Enhanced prompt variants
```

### Key Result Files

**Detailed Results** (`baseline_async_results_detailed_*.json`):
- Individual predictions for every model-prompt-input combination
- Execution metadata (response times, token usage)
- Error breakdowns and accuracy metrics

**Context Enhancement Results** (Phase 2 files):
- Best variant selection rationale
- Context example selection with diversity metrics
- Prefix vs suffix ordering comparison

### Analysis Ready
The experiment generates complete datasets ready for analysis. **Next step**: Create analysis notebook (`analysis.ipynb`) to visualize:
- Model robustness profiles across prompt dimensions
- Consistency patterns and failure modes
- Context enhancement effectiveness
- Prompt engineering insights

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

# Local model support (Ollama)
--models qwen3:4b
```

### Experiment Parameters
- `--n-samples`: Number of test samples (default: from config)
- `--run-phase2`: Include context enhancement analysis
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## 📦 Project Structure

```
sentiment-LLM-robust-eval/
├── scripts/
│   └── run_experiment.py              # 🚀 Main experiment entry point
├── deep_learning_final_assignment/    # Core framework
│   ├── core/
│   │   ├── models/                    # Model interfaces (OpenAI, Ollama)
│   │   ├── prompts/                   # 16-variant prompt system
│   │   ├── evaluation/                # Metrics and consistency calculation
│   │   ├── context_enhancement/       # Phase 2 context system
│   │   └── utils/                     # Logging and utilities
│   └── ...
├── tests/                             # 208+ comprehensive tests
├── config/                            # Model and experiment configuration
├── results/                           # 📊 Experiment outputs (timestamped)
└── analysis.ipynb                     # 📈 Analysis notebook (to be created)
```

## 🎯 Research Focus

**Primary Comparison**: GPT-4.1 vs GPT-4o-mini robustness across systematic prompt variations

**Evaluation Metrics**:
- **Custom Accuracy**: Polarity-weighted encoding with penalty for extreme errors
- **Consistency Score**: Prediction agreement across prompt variants
- **Weighted Index**: Combined accuracy + consistency ranking (70% / 30%)

### **Detailed Evaluation System**

#### **1. Polarity-Weighted Encoding**
We encode sentiment labels to emphasize polarity mistakes over intensity mistakes:

```
Very Negative = -3
Negative      = -2  
Neutral       =  0
Positive      = +2
Very Positive = +3
```

#### **2. MSE-Based Custom Accuracy**
**Formula**: `sklearn.metrics.mean_squared_error(actual_encodings, predicted_encodings)`

**Penalty Structure**:
- **Adjacent Errors** (1 point): Negative ↔ Very Negative, Positive ↔ Very Positive
- **Base→Neutral Errors** (4 points): Negative ↔ Neutral, Positive ↔ Neutral  
- **Cross-Polarity Errors** (16 points): Negative ↔ Positive
- **Extreme Errors** (36 points): Very Negative ↔ Very Positive

**Rationale**: Polarity mistakes (positive/negative confusion) are penalized more heavily than intensity mistakes, reflecting real-world deployment priorities.

#### **3. Consistency Calculation (Within-Model)**

**Per-Input Consistency**:
1. Collect predictions from all 16 variants for each model
2. Calculate prediction distribution: `{label: count/16 for label in predictions}`
3. Input consistency = `max(distribution.values())` (highest agreement percentage)

**Model Consistency**: 
- Average per-input consistency scores across all test inputs
- **Example**: If 12/16 variants predict "Positive" for input, consistency = 0.75

**Weighted Index**: `0.7 × variant_accuracy + 0.3 × model_consistency`
- Each variant uses its own accuracy + its model's overall consistency
- Ensures within-model comparison while rewarding both accuracy and robustness

### **Expected Research Insights**
- Which prompt formulations are most robust across models?
- How do different models respond to systematic prompt variations?
- Does context enhancement improve consistency and accuracy?
- What are the optimal prompt engineering strategies for deployment?
- Which dimensional changes cause the most prediction instability?

## 🔧 Development

### Dependencies
- **ML/NLP**: transformers, datasets, torch, scikit-learn
- **API Integration**: openai, ollama  
- **Evaluation**: pandas, sentence-transformers
- **Testing**: pytest, pytest-cov, pytest-asyncio

### Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`  
3. Run tests: `poetry run pytest`
4. Submit pull request

---

**Status**: ✅ Experiment framework complete and tested  
**Next Step**: Create analysis notebook to evaluate experiment results  
**Course**: CEU MSBA Deep Learning (Prof. Eduardo Arino de la Rubia)  
**Deadline**: May 25, 2025