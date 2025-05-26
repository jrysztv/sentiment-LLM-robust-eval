# Configuration Guide

This directory contains configuration files for the Deep Learning Final Assignment project.

## Configuration Files

### `experiment_config.json`
Main configuration file containing all experiment parameters:

- **Models**: OpenAI and Ollama model settings
- **Experiment**: Dataset and sampling parameters  
- **Evaluation**: Metrics and scoring weights
- **Output**: Results directory and logging settings

## Model Configuration

### OpenAI Model (GPT-4.1)

The configuration uses `gpt-4.1` which is the latest GPT-4.1 model in the OpenAI API:

```json
{
  "models": {
    "openai": {
      "model_name": "gpt-4.1",
      "temperature": 0.1,
      "max_tokens": 150,
      "json_mode": true
    }
  }
}
```

**Alternative models:**
- `gpt-4o-mini`: Faster and cheaper alternative
- `gpt-4o`: Previous generation (if preferred)

### Ollama Model (Qwen)

The configuration uses `qwen3:4b` which is the latest available Qwen3 model:

```json
{
  "models": {
    "ollama": {
      "model_name": "qwen3:4b",
      "temperature": 0.1,
      "max_tokens": 150,
      "host": "http://localhost:11434",
      "auto_pull": true
    }
  }
}
```

**Note:** `qwen3:4b` is the latest available Qwen3 model, which is the intended model for this assignment.

## Setup Instructions

### 1. OpenAI Setup

1. Get an API key from [OpenAI Platform](https://platform.openai.com/)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. The model `gpt-4.1` should be available with your API key

### 2. Ollama Setup

#### Installation

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai/download](https://ollama.ai/download)

#### Model Download

Use our setup script:
```bash
python scripts/setup_ollama.py
```

Or manually:
```bash
# Start Ollama server
ollama serve

# Download the model (in another terminal)
ollama pull qwen3:4b
```

### 3. Verification

Run the setup verification:
```bash
python scripts/setup_ollama.py
```

## Configuration Usage

### Using Default Configuration

```bash
python scripts/run_baseline.py
```

### Using Custom Configuration

```bash
python scripts/run_baseline.py --config path/to/your/config.json
```

### Command Line Overrides

```bash
# Override specific settings
python scripts/run_baseline.py --openai-model gpt-4o-mini --n-samples 100

# Override Ollama model
python scripts/run_baseline.py --ollama-model qwen3:4b
```

## Model Recommendations

### For Development/Testing
- OpenAI: `gpt-4o-mini` (faster, cheaper)
- Ollama: `qwen3:1.7b` (smaller, faster alternative)

### For Final Results
- OpenAI: `gpt-4.1` (best performance)
- Ollama: `qwen3:4b` (good performance)

## Troubleshooting

### OpenAI Issues
- **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
- **Model Not Available**: Check your OpenAI plan supports GPT-4
- **Rate Limits**: Reduce `n_samples` or add delays

### Ollama Issues
- **Server Not Running**: Run `ollama serve` in a terminal
- **Model Not Found**: Run `python scripts/setup_ollama.py`
- **Download Failed**: Check internet connection and disk space
- **Port Conflict**: Change `host` in config to use different port

### Configuration Issues
- **File Not Found**: Ensure `config/experiment_config.json` exists
- **JSON Errors**: Validate JSON syntax
- **Permission Errors**: Check file permissions

## Configuration Schema

The configuration file supports the following structure:

```json
{
  "models": {
    "openai": {
      "model_name": "string",
      "temperature": "float (0.0-2.0)",
      "max_tokens": "integer",
      "json_mode": "boolean",
      "api_key_env": "string"
    },
    "ollama": {
      "model_name": "string", 
      "temperature": "float (0.0-2.0)",
      "max_tokens": "integer",
      "host": "string (URL)",
      "auto_pull": "boolean"
    }
  },
  "experiment": {
    "task_name": "string",
    "data_split": "train|validation|test",
    "n_samples": "integer",
    "balanced_sampling": "boolean",
    "random_seed": "integer"
  },
  "evaluation": {
    "weighted_index_weights": {
      "accuracy": "float (0.0-1.0)",
      "consistency": "float (0.0-1.0)"
    }
  },
  "output": {
    "results_dir": "string (path)",
    "log_level": "DEBUG|INFO|WARNING|ERROR",
    "save_intermediate": "boolean"
  }
}
``` 