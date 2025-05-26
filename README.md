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

**Expected Insights**:
- Which prompt formulations are most robust?
- How do different models respond to prompt variations?
- Does context enhancement improve consistency?
- What are the optimal prompt engineering strategies?

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