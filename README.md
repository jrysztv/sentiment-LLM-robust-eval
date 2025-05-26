# CEU Deep Learning Final Assignment: Prompt Perturbation Robustness Testing

A comprehensive robustness testing framework for foundation models, evaluating how model performance varies across systematic prompt variations. Built with Python, featuring modern development practices, and robust testing infrastructure.

## 🎯 Project Overview

This project implements systematic robustness testing for large language models through prompt perturbation analysis. We evaluate model consistency and accuracy across 16 different prompt formulations to understand how sensitive foundation models are to variations in prompt phrasing.

### ✅ **Phase 1.5 COMPLETED** - Enhanced Model Flexibility & Consistency Analysis
- **Flexible Model Selection**: Runtime model switching via CLI (`--models gpt-4.1,gpt-4o-mini`)
- **Corrected Consistency Calculation**: Per-input consistency as max(prediction_distribution), model consistency as average
- **Enhanced Data Persistence**: Complete audit trails with formatted prompts, raw responses, and execution metadata
- **Unicode-Safe Logging**: Full emoji support with Windows compatibility
- **Comprehensive Testing**: 19 tests covering enhanced functionality with 100% pass rate

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- OpenAI API key (for GPT models)
- Optional: Ollama (for local models)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deep-learning-final-assignment
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

4. Activate the virtual environment:
```bash
poetry shell
```

### Running Robustness Tests

#### Phase 1.5 Enhanced Baseline Testing

**Primary Research Focus: GPT-4.1 vs GPT-4o-mini comparison**
```bash
poetry run python scripts/run_baseline_enhanced.py --config config/enhanced_experiment_config.json --models gpt-4.1,gpt-4o-mini --n-samples 50
```

**Quick Testing with GPT-4o-mini only**
```bash
poetry run python scripts/run_baseline_enhanced.py --config config/enhanced_experiment_config.json --models gpt-4o-mini --n-samples 10
```

**All configuration options**
```bash
poetry run python scripts/run_baseline_enhanced.py --help
```

#### Output Files
- `detailed_execution_data_TIMESTAMP.json` - Complete audit trail with per-input analysis
- `baseline_results_detailed_TIMESTAMP.json` - Enhanced results with dimensional metadata  
- `baseline_results_summary_TIMESTAMP.json` - Model performance summaries

## 🧪 Testing

This project uses pytest for comprehensive testing with coverage reporting.

### Run All Tests
```bash
poetry run pytest
```

### Run Tests with Verbose Output
```bash
poetry run pytest -v
```

### Run Only Unit Tests
```bash
poetry run pytest -m unit
```

### Run Only Integration Tests
```bash
poetry run pytest -m integration
```

### Skip Slow Tests
```bash
poetry run pytest -m "not slow"
```

### Generate Coverage Report
```bash
poetry run pytest --cov=deep_learning_final_assignment --cov-report=html
```

### Test Configuration

The project is configured with:
- **Coverage threshold**: 80% minimum
- **Test discovery**: Automatic discovery of `test_*.py` and `*_test.py` files
- **Markers**: `unit`, `integration`, `slow`, `requires_api`
- **HTML coverage reports**: Generated in `htmlcov/` directory

## 📁 Project Structure

```
deep-learning-final-assignment/
├── deep_learning_final_assignment/   # Main package
│   ├── __init__.py                   # Package initialization
│   └── ...                          # Module files
├── tests/                            # Test suite
│   ├── __init__.py                   # Test package init
│   ├── conftest.py                   # Pytest configuration & fixtures
│   ├── test_package.py               # Basic package tests
│   └── data/                         # Test data directory
├── analysis.ipynb                    # Jupyter notebook for analysis
├── pyproject.toml                    # Project configuration
├── poetry.lock                       # Dependency lock file
└── README.md                         # This file
```

## 🛠 Development

### Adding Dependencies

For runtime dependencies:
```bash
poetry add <package-name>
```

For development dependencies:
```bash
poetry add --group dev <package-name>
```

### Code Quality

This project follows Python best practices:
- Type hints where appropriate
- Comprehensive test coverage (80%+ required)
- Clear documentation
- Modular design

## 📦 Dependencies

### Runtime Dependencies
- **Core ML/NLP**: transformers, datasets, torch, scikit-learn, sentence-transformers
- **API Integration**: openai, ollama
- **Evaluation**: rouge-score, pandas, python-dotenv, jupyter
- **PEFT**: peft, accelerate
- **Testing**: pytest, pytest-cov, pytest-mock, pytest-asyncio

### Development Dependencies
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-asyncio**: Async testing support

## 🎯 Project Goals

This is a deep learning final assignment project focusing on:
- **Robustness Testing**: Evaluate how foundation models respond to systematic variations in prompt phrasing
- **Consistency Analysis**: Measure prediction consistency across 16 prompt variants using 4 systematic dimensions
- **Model Comparison**: Compare GPT-4.1 vs GPT-4o-mini performance across prompt variations
- **Enhanced Evaluation**: Custom accuracy metrics with polarity-weighted encoding and comprehensive data persistence

## 📊 Analysis

The project provides comprehensive analysis capabilities:
- **Dimensional Analysis**: Compare formal vs casual, imperative vs question, task-first vs text-first, synonym variations
- **Consistency Patterns**: Per-input and model-level consistency calculations
- **Performance Metrics**: Custom accuracy with penalty system for polarity mistakes
- **Complete Audit Trails**: Full execution data with formatted prompts, raw responses, and metadata

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `poetry run pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

[Add license information]

## 👨‍💻 Author

**Istvan Peter Jaray**  
Email: istvanpeterjaray@gmail.com