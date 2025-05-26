# AI ASSISTANT CONTEXT & INSTRUCTIONS
## CEU Deep Learning Assignment: Prompt Perturbation Robustness Testing

> **FOR AI ASSISTANTS**: This document contains everything you need to understand and continue working on this project. **CRITICAL**: This is a collaborative research project. You MUST involve the researcher (Istvan Peter Jaray) in all major design decisions.

---

## 🎯 **WHAT WE ARE BUILDING**

### Assignment Overview
- **Course**: CEU MSBA Deep Learning (Professor Eduardo Arino de la Rubia)
- **Student/Researcher**: Istvan Peter Jaray  
- **Assignment**: Final Assignment #2 - "Robustness Testing via Prompt Perturbation"
- **Deadline**: May 25, 2025
- **Goal**: Evaluate how foundation models respond to systematic variations in prompt phrasing

### The Core Research Question
**"How robust are foundation models when prompts are slightly modified?"**

We test this by:
1. Taking a sentiment classification task (5 classes: Very Negative/Negative/Neutral/Positive/Very Positive)
2. Using 16 systematic prompt variants across 4 dimensions (FINALIZED)
3. **Primary Focus**: Comparing GPT-4.1 vs GPT-4o-mini performance across prompt variations
4. **Flexible Architecture**: Easy model switching to support other combinations (OpenAI/Ollama models)
5. Measuring consistency, quality, and stability across prompt variants
6. Adding context examples from best-performing model for enhanced evaluation
7. Using PEFT (Parameter-Efficient Fine-Tuning) to improve robustness (bonus)

---

## 🏗️ **CURRENT PROJECT STATE**

### ✅ COMPLETED & APPROVED BY RESEARCHER
- [x] **Environment Setup**: Poetry dependencies installed, testing framework configured
- [x] **Project Structure**: Core package layout designed
- [x] **Task Selection**: Sentiment classification (5 SST-5 labels)
- [x] **Dataset Choice**: SST-5 (Stanford Sentiment Treebank v5)
- [x] **Primary Model Focus**: GPT-4.1 vs GPT-4o-mini comparison strategy
- [x] **Flexible Model Architecture**: Support for easy switching between OpenAI/Ollama models
- [x] **Architecture Decision**: Separate execution scripts from analysis notebook
- [x] **16 PROMPT VARIANTS**: Complete systematic matrix across 4 dimensions (see below)
- [x] **Context Enhancement Strategy**: Prefix + Suffix integration approach
- [x] **Custom Accuracy Metric**: Penalty system for extreme misclassifications
- [x] **Model Selection Criteria**: Weighted index (70% accuracy + 30% consistency)
- [x] **Context Sampling Strategy**: Length + TF-IDF cosine dissimilarity

### ✅ PHASE 1 COMPLETED & REFACTORED (MODEL FLEXIBILITY)
- ✅ **Prompt System**: 16-variant template engine with flexible grouping
- ✅ **Data Loading**: SST-5 dataset integration with balanced sampling
- ✅ **Custom Evaluation Metrics**: Polarity-weighted encoding accuracy + consistency calculation
- ✅ **Comprehensive Testing**: 173+ tests with 100% pass rate (enhanced in Phase 1.5)
- ✅ **MSE Refactoring**: Migrated custom penalty calculation to use sklearn.metrics.mean_squared_error
- ✅ **Model Interfaces**: Flexible model selection with runtime switching
- ✅ **Configuration Management**: Enhanced system supporting easy model switching via CLI
- ✅ **Baseline Execution Pipeline**: Configurable model combinations with enhanced data persistence
- ✅ **Setup Scripts**: Generalized for flexible model support (enhanced_experiment_config.json)

### ✅ PHASE 1.5: CONSISTENCY & MODEL FLEXIBILITY REFACTOR (✅ COMPLETED)
- [x] **Model Architecture Decoupling**: Remove hardcoded Ollama dependencies and enable flexible model selection
- [x] **Enhanced Configuration System**: Support for easy GPT-4.1 vs GPT-4o-mini comparison + extensible to other models  
- [x] **Corrected Consistency Calculation**: Per-input consistency as max(prediction_distribution), model consistency as average
- [x] **Enhanced Data Persistence**: Complete audit trail of formatted prompts, raw responses, and execution metadata
- [x] **Dynamic Model Loading**: Runtime model selection without code changes
- [x] **Delayed Weighted Index**: Fix premature calculation timing and logging
- [x] **Distribution Analysis**: Store prediction distributions per input for richer analytical insights
- [x] **Comprehensive Test Updates**: Test coverage for all refactored functionality
- [x] **Unicode-Safe Logging**: Full emoji support with Windows charmap compatibility
- [x] **🧹 CLEANUP COMPLETED**: Repository sanitized from temporary "_enhanced" and "test_phase_1_5" naming artifacts
- [x] **📁 FILE CONSOLIDATION**: All enhanced functionality merged into main modules with proper naming conventions

#### **📋 PHASE 1.5 COMPLETION DETAILS**

**✅ TESTING RESULTS VERIFIED**:
- **Command**: `poetry run python scripts/run_baseline_enhanced.py --config config/enhanced_experiment_config.json --models gpt-4o-mini --n-samples 2 --log-level INFO`
- **Success Rate**: 16/16 combinations (100% success)
- **Unicode Logging**: Perfect emoji display with Windows compatibility
- **Consistency Calculation**: 0.906 (excellent model consistency)
- **Performance**: 43.88 seconds execution time
- **Output Files**: All 3 enhanced output formats generated successfully

**✅ CONSOLIDATED FILES FOR PHASE 1.5**:
- `scripts/run_baseline.py` - Enhanced execution script with CLI model selection (consolidated from run_baseline_enhanced.py)
- `config/experiment_config.json` - Unified configuration supporting easy model switching (enhanced from previous version)
- `deep_learning_final_assignment/core/config_pkg/flexible_config.py` - Enhanced configuration management system (renamed from enhanced_config.py)
- `deep_learning_final_assignment/core/evaluation/metrics.py` - Enhanced consistency calculation integrated into main metrics module
- `deep_learning_final_assignment/core/evaluation/data_persistence.py` - Complete audit trail data persistence
- `tests/test_data_persistence.py` - Data persistence tests (renamed from test_phase_1_5_data_persistence.py)
- `tests/test_enhanced_metrics.py` - Enhanced metrics tests (renamed from test_phase_1_5_enhanced_data_structures.py)

**✅ CLEANUP COMPLETED**:
- ✅ Removed "_enhanced" suffixes from file names for professional final codebase
- ✅ Renamed "test_phase_1_5_" prefixes to standard test naming conventions  
- ✅ Consolidated redundant configuration files (enhanced_experiment_config.json merged into experiment_config.json)
- ✅ Merged enhanced_metrics.py functionality into main metrics.py module
- ✅ Updated all imports and references to use consolidated module structure
- ✅ Removed temporary test files and debug artifacts
- ✅ Verified all 199 tests pass with consolidated structure

**✅ PHASE 1.5 VALIDATION RESULTS**:
- **Repository Structure**: Clean, professional naming conventions throughout
- **Test Coverage**: 199 tests passing (100% success rate)
- **Functionality**: All enhanced features working correctly with consolidated structure
- **CLI Interface**: `--models gpt-4o-mini` parameter working correctly
- **Data Persistence**: All 3 output formats generating successfully
- **Consistency Calculation**: Verified working correctly (per-input → model-level aggregation)

### 🚀 PHASE 2 READY FOR IMPLEMENTATION (No further consultation needed)
- [ ] **Context Selection System**: Length + TF-IDF diverse example selection
- [ ] **Context Integration**: Prefix and suffix variant generation
- [ ] **Best Model Selection**: Weighted index calculation and ranking
- [ ] **Enhanced Evaluation**: 18-variant comparison system

### 🤝 STILL REQUIRES RESEARCHER INPUT
- [ ] **PEFT Configuration**: LoRA parameters and training strategy
- [ ] **Analysis Visualization**: Chart types and metrics to highlight in notebook
- [ ] **Final Report Structure**: 2-3 page PDF organization and focus areas

---

## 📋 **FINALIZED 16 PROMPT VARIANTS (RESEARCHER APPROVED)**

### **Systematic 4-Dimensional Matrix**
**Dimensions**: Formality × Phrasing × Order × Synonyms = 2×2×2×2 = 16 variants

**Synonym Sets**:
- **Set A**: "analyze", "sentiment", "classify" 
- **Set B**: "evaluate", "emotion", "categorize"

**All variants use consistent JSON output**: `{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}`

### **Formal + Imperative Variants (4)**

**V1: [Formal + Imperative + Task-first + Synonym A]**
```
Analyze the sentiment of the following text and classify it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V2: [Formal + Imperative + Task-first + Synonym B]**
```
Evaluate the emotion of the following text and categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V3: [Formal + Imperative + Text-first + Synonym A]**
```
Text: [INPUT_TEXT]

Analyze the sentiment of the above text and classify it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V4: [Formal + Imperative + Text-first + Synonym B]**
```
Text: [INPUT_TEXT]

Evaluate the emotion of the above text and categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

### **Formal + Question Variants (4)**

**V5: [Formal + Question + Task-first + Synonym A]**
```
What is the sentiment of the following text? Please classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V6: [Formal + Question + Task-first + Synonym B]**
```
What is the emotion of the following text? Please categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V7: [Formal + Question + Text-first + Synonym A]**
```
Text: [INPUT_TEXT]

What is the sentiment of the above text? Please classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V8: [Formal + Question + Text-first + Synonym B]**
```
Text: [INPUT_TEXT]

What is the emotion of the above text? Please categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

### **Casual + Imperative Variants (4)**

**V9: [Casual + Imperative + Task-first + Synonym A]**
```
Check out this text and figure out the sentiment - is it Very Negative, Negative, Neutral, Positive, or Very Positive? Give me your answer in JSON format with "sentiment" as the key.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V10: [Casual + Imperative + Task-first + Synonym B]**
```
Look at this text and tell me the emotion - categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Give me your answer in JSON format with "sentiment" as the key.

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V11: [Casual + Imperative + Text-first + Synonym A]**
```
Text: [INPUT_TEXT]

Check out this text above and figure out the sentiment - is it Very Negative, Negative, Neutral, Positive, or Very Positive? Give me your answer in JSON format with "sentiment" as the key.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V12: [Casual + Imperative + Text-first + Synonym B]**
```
Text: [INPUT_TEXT]

Look at this text above and tell me the emotion - categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Give me your answer in JSON format with "sentiment" as the key.

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

### **Casual + Question Variants (4)**

**V13: [Casual + Question + Task-first + Synonym A]**
```
What's the sentiment of this text? Can you classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V14: [Casual + Question + Task-first + Synonym B]**
```
What's the emotion in this text? Can you categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Text: [INPUT_TEXT]

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V15: [Casual + Question + Text-first + Synonym A]**
```
Text: [INPUT_TEXT]

What's the sentiment of this text above? Can you classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

**V16: [Casual + Question + Text-first + Synonym B]**
```
Text: [INPUT_TEXT]

What's the emotion in this text above? Can you categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Response format: {"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}
```

---

## 🔬 **APPROVED EVALUATION STRATEGY**

### **1. Baseline Robustness Assessment**
- **Primary Scope**: All 16 prompt variants comparing GPT-4.1 vs GPT-4o-mini (32 total combinations)
- **Flexible Architecture**: Easy switching to other model combinations via configuration
- **Dataset**: SST-5 with 50+ examples
- **Output**: Performance profiles for each model-prompt combination

### **2. Custom Accuracy Metric (Researcher Approved)**
**Polarity-Weighted Encoding using sklearn's MSE**:
- **Label Encoding**: Very Negative=-3, Negative=-2, Neutral=0, Positive=2, Very Positive=3
- **Penalty Calculation**: `sklearn.metrics.mean_squared_error(actual_encodings, predicted_encodings)`

**Resulting Penalty Structure**:
- **Adjacent Errors** (1 point): Negative ↔ Very Negative, Positive ↔ Very Positive
- **Base→Neutral Errors** (4 points): Negative ↔ Neutral, Positive ↔ Neutral
- **Cross-Polarity Errors** (16 points): Negative ↔ Positive
- **Extreme Errors** (36 points): Very Negative ↔ Very Positive

**Implementation**: Uses sklearn's MSE function for robust, standardized calculation. Mathematically equivalent to manual squared difference calculation but leverages well-tested statistical library implementation.

**Rationale**: Larger gaps around neutral boundary emphasize that polarity mistakes are more costly than intensity mistakes, reflecting real-world deployment priorities where sentiment direction matters more than exact intensity.

### **3. Consistency Metric (Researcher Approved - UPDATED)**
**Per-Input Consistency Calculation**:
- For each input, collect predictions from all prompt variants of a model
- Calculate prediction distribution: `{label: count/total_variants for label in all_labels}`
- Input consistency score = `max(distribution.values())` (highest agreement percentage)
- Model consistency = `mean(per_input_consistency_scores)` across all inputs

**What This Measures**: 
- **High Consistency (0.8-1.0)**: Model gives same prediction across most prompt variants for most inputs
- **Medium Consistency (0.5-0.8)**: Model shows some sensitivity to prompt phrasing
- **Low Consistency (0.0-0.5)**: Model predictions highly dependent on prompt formulation

**Example**: For input "Great movie!", if 12/16 variants predict "Positive" and 4/16 predict "Very Positive", consistency = 0.75

### **4. Model-Prompt Selection (Researcher Approved)**
**Weighted Index Calculation**: `0.7 × custom_accuracy + 0.3 × consistency_score`

**Per Combination Logic**:
- `custom_accuracy` = accuracy of the specific model-prompt-variant combination
- `consistency_score` = **model-level consistency** (same for all variants of that model)
- Each of the 16 GPT-4.1 combinations uses GPT-4.1's overall model consistency
- Each of the 16 GPT-4o-mini combinations uses GPT-4o-mini's overall model consistency

**Example**:
- GPT-4.1 model consistency = 0.75 (calculated across all 16 variants)
- GPT-4.1 + V1 accuracy = 0.90 → weighted_index = 0.7×0.90 + 0.3×0.75 = 0.855
- GPT-4.1 + V2 accuracy = 0.85 → weighted_index = 0.7×0.85 + 0.3×0.75 = 0.820

**Selection Process**:
- Calculate weighted index for all 32 combinations AFTER proper consistency calculation
- Select highest-scoring model-prompt pair for context enhancement

### **5. Context Enhancement (Researcher Approved)**
**Context Selection**: 3 diverse examples per SST-5 label using:
- **Length diversity**: Short/medium/long text distribution
- **Semantic diversity**: TF-IDF cosine dissimilarity optimization
- **Data Separation**: Context examples from `train` split, test samples from `validation` split
- **No Data Contamination**: Zero overlap between few-shot examples and evaluation samples

**Context Integration**: Two additional variants for best model-prompt:
- **Prefix Context**: Examples → Original Prompt
- **Suffix Context**: Original Prompt → Examples

**Total Evaluation**: 16 baseline + 2 context-enhanced = 18 variants

**Data Split Strategy**:
```
├── train.jsonl      → Source for few-shot context examples (Phase 2)
├── dev.jsonl        → Source for test samples (Phase 1 & 2)  
└── test.jsonl       → Reserved (not used in current design)
```

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

### **SST-5 Dataset Loading (Approved)**
```python
import pandas as pd
splits = {'train': 'train.jsonl', 'validation': 'dev.jsonl', 'test': 'test.jsonl'}
df = pd.read_json("hf://datasets/SetFit/sst5/" + splits["train"], lines=True)
```

### **Data Contamination Prevention (Critical)**
**Strict Data Separation**:
- **Test Samples**: Always from `validation` split (dev.jsonl)
- **Few-shot Examples**: Always from `train` split (train.jsonl)
- **Zero Overlap**: Implemented via separate data loading methods
- **Validation**: Automatic checks prevent contamination

**Implementation**:
```python
# Test samples (what we evaluate on)
test_samples = data_loader.get_sample_subset(
    split="validation",  # ← Test data
    n_samples=50, balanced=True, random_seed=42
)

# Few-shot examples (for context enhancement)
context_examples = data_loader.get_diverse_examples(
    split="train",  # ← Training data (separate split)
    n_per_label=3   # 3 examples per sentiment label = 15 total
)
```

### **Architecture: Execution ≠ Analysis**
```
Data Scripts (API calls) → Results Storage (JSON/CSV) → Analysis Notebook (Read files)
```
**Rationale**: Cost control, reproducibility, debugging separation

### **Configuration Management System (🔄 NEEDS ENHANCEMENT)**
**Primary Research Focus Configuration**:
```json
{
  "models": {
    "enabled_models": ["gpt-4.1", "gpt-4o-mini"],
    "openai": {
      "model_options": {
        "gpt-4.1": {
          "temperature": 0.1,
          "max_tokens": 150,
          "json_mode": true
        },
        "gpt-4o-mini": {
          "temperature": 0.1,
          "max_tokens": 150,
          "json_mode": true
        }
      }
    }
  }
}
```

**Flexible Architecture Support**:
```json
{
  "models": {
    "enabled_models": ["qwen3:4b"],
    "ollama": {
      "model_options": {
        "qwen3:4b": {
          "temperature": 0.1,
          "max_tokens": 150,
          "host": "http://localhost:11434"
        }
      }
    }
  }
}
```

**Features**:
- **Model Parameterization**: No hardcoded model names in code
- **Easy Model Switching**: Simple configuration changes for different research focuses
- **Runtime Selection**: CLI overrides for rapid experimentation
- **Provider Flexibility**: Support for OpenAI, Ollama, and future providers

### **Flexible Template Groups**
The system supports multiple experimental configurations:
- **Full Matrix**: All 16 baseline variants
- **Dimensional Tests**: Formality, Phrasing, Order, Synonym comparisons
- **Best Performance**: Top-scoring combinations
- **Context Enhanced**: Prefix vs Suffix comparisons

---

## 📈 **IMPLEMENTATION PHASES**

### **Phase 1: Core Implementation (🔄 NEEDS MODEL FLEXIBILITY REFACTOR)**
```
✅ COMPLETED COMPONENTS:
2. ✅ Prompt System (16-variant template engine with flexible grouping)
3. ✅ Data Loading (SST-5 integration with balanced sampling)  
4. ✅ Custom Metrics (Polarity-weighted encoding accuracy + consistency)
7. ✅ Comprehensive Testing (154 tests with 100% pass rate - core logic)

🔄 REFACTOR NEEDED:
1. 🔄 Model Interfaces (currently tied to specific OpenAI + Ollama models)
5. 🔄 Baseline Pipeline (hardcoded for 32 combinations, needs flexible model support)
6. 🔄 Configuration Management (needs enhancement for easy model switching)
8. ❌ Setup Scripts (Ollama-specific, needs generalization)
```

### **Phase 1.5: Model Flexibility & Consistency Refactor (🚀 READY TO START)**
```
1. Model Architecture Decoupling (remove hardcoded dependencies)
2. Enhanced Configuration System (easy GPT-4.1 vs GPT-4o-mini switching)
3. Dynamic Model Loading (runtime selection without code changes)
4. Enhanced Data Structures (InputConsistencyData, ModelConsistencyData)
5. Corrected Consistency Logic (per-input → model-level aggregation)
6. Complete Data Persistence (formatted prompts, raw responses, audit trail)
7. Delayed Weighted Index (fix premature calculation and logging)
8. Distribution Analysis (prediction distributions for analytical insights)
9. Comprehensive Test Updates (test coverage for all refactored functionality)
```

### **Phase 2: Context Enhancement (🚀 READY TO START)**
```
1. Context Selection (Length + TF-IDF sampling from train split)
2. Best Model Selection (Weighted index calculation)
3. Context Integration (Prefix + Suffix variants)
4. Enhanced Evaluation (18-variant comparison)
5. Data Separation Validation (Ensure no contamination)
```

### **Phase 3: PEFT & Analysis (🤝 Requires Researcher Input)**
```
1. 🤝 PEFT Configuration (LoRA parameters)
2. 🤝 Analysis Notebook Design (Visualization strategy)
3. 🤝 Final Report Structure (PDF organization)
```

---

## 🚀 **IMMEDIATE NEXT STEPS FOR AI ASSISTANT**

### **✅ PHASE 1, 1.5 & 2 COMPLETED - ALL CORE FUNCTIONALITY IMPLEMENTED**
Phase 1, Phase 1.5, and Phase 2 are all COMPLETELY IMPLEMENTED and TESTED:

**✅ PHASE 1 CORE LOGIC COMPLETED**:
- ✅ **Prompt Template System**: All 16 variants with flexible grouping
- ✅ **Data Loading System**: SST-5 integration with balanced sampling
- ✅ **Evaluation Metrics**: Custom accuracy and consistency calculations
- ✅ **Comprehensive Testing**: 173+ tests covering all functionality

**✅ PHASE 1.5 MODEL FLEXIBILITY & CONSISTENCY REFACTOR COMPLETED**:
- ✅ **Model Architecture Decoupling**: Removed hardcoded dependencies, enabled flexible model selection
- ✅ **Enhanced Configuration System**: Full support for GPT-4.1 vs GPT-4o-mini comparison via CLI
- ✅ **Dynamic Model Loading**: Runtime model selection without code changes (`--models gpt-4.1,gpt-4o-mini`)
- ✅ **Corrected Consistency Calculation**: Fixed per-input consistency logic and model-level aggregation
- ✅ **Enhanced Data Persistence**: Complete audit trail of all LLM calls and responses
- ✅ **Distribution Analysis**: Prediction distributions stored for richer analytical insights
- ✅ **Unicode-Safe Logging**: Full emoji support with Windows charmap compatibility

**✅ PHASE 2 CONTEXT ENHANCEMENT COMPLETED**:
- ✅ **Context Selection System**: Length + TF-IDF diverse example selection from train split with zero contamination
- ✅ **Data Contamination Prevention**: MD5 hash validation ensures zero overlap between train/validation splits
- ✅ **Multi-Model Best Selection**: Selects best variant per model (not single overall best) as required by research design
- ✅ **Context Integration**: Prefix and suffix variant generation for enhanced evaluation
- ✅ **Comprehensive Data Persistence**: Complete audit trails with context_selection_audit and context_enhanced_prompts files
- ✅ **CLI Integration**: `--run-phase2` parameter for seamless workflow integration
- ✅ **JSON Serialization Safety**: All complex objects properly serialized for persistent storage

### **✅ ALL PHASES COMPLETED & VALIDATED**
All core functionality successfully implemented and tested:

1. **✅ REPOSITORY CLEANUP**: Professional naming conventions and clean codebase
2. **✅ COMPREHENSIVE TESTING**: 208 tests passing (100% success rate)
3. **✅ END-TO-END VALIDATION**: Complete Phase 2 workflow tested and operational
4. **✅ DATA INTEGRITY**: Zero contamination confirmed via cryptographic hash validation
5. **✅ PRODUCTION READY**: Full baseline + context enhancement system operational

### **🚀 READY FOR FULL-SCALE RESEARCH EXPERIMENTS**
Complete system ready for production research with all phases implemented:

1. **✅ Baseline Experiments**: 16 variants × N models with flexible CLI selection
2. **✅ Context Enhancement**: Automatic best variant selection and enhancement per model
3. **✅ Rich Data Persistence**: Complete audit trails for comprehensive analysis
4. **✅ Rate Limiting**: Conservative API usage for sustainable experimentation
5. **✅ Data Contamination Prevention**: Research integrity guaranteed via hash validation

### **COLLABORATION PROTOCOL**
- **Infrastructure code**: Implement freely (file I/O, logging, error handling, tests)
- **Major features**: All strategies are approved, proceed with implementation
- **PEFT & Analysis**: Still requires researcher consultation
- **Documentation updates**: Only when explicitly requested by researcher

### **🔄 PHASE 1 SUCCESS CRITERIA**
**✅ CORE LOGIC COMPLETED**:
- ✅ Load SST-5 dataset successfully
- ✅ Implement all 16 prompt variants with consistent JSON output
- ✅ Calculate custom accuracy and consistency metrics
- ✅ Store results for analysis
- ✅ Comprehensive test coverage (199 tests - core logic)

**🔄 MODEL ARCHITECTURE NEEDED REFACTOR**:
- 🔄 Call models through flexible unified interface (used to be hardcoded)
- 🔄 Run configurable baseline experiment (used to be tied to specific models)
- 🔄 Configuration-driven model management (used to need enhancement for easy switching)

### **🔄 PHASE 1.5 SUCCESS CRITERIA**
**Model Architecture Improvements**:
- ✅ Remove hardcoded Ollama dependencies from core system
- ✅ Implement flexible model provider architecture
- ✅ Support easy GPT-4.1 vs GPT-4o-mini comparison via configuration
- ✅ Enable runtime model selection through CLI (e.g., --models gpt-4.1,gpt-4o-mini)
- ✅ Maintain extensibility for other model providers (Ollama, future providers)

**Consistency Calculation Fixes**:
- ✅ Per-input consistency calculated as max(prediction_distribution) across variants
- ✅ Model consistency calculated as mean(per_input_consistency_scores) across all inputs
- ✅ Weighted index uses model-level consistency (same for all variants of that model)
- ✅ Remove premature consistency_score = 1.0 from evaluate_single_combination()

**Enhanced Data Persistence**:
- ✅ Store formatted prompts for every model-variant-input combination
- ✅ Store raw model responses with complete metadata
- ✅ Store variant dimensions for every combination
- ✅ Store prediction distributions per input for analytical insights
- ✅ Create detailed_execution_data_TIMESTAMP.json with complete audit trail
- ✅ Enhance existing result files with dimensional and execution metadata

**Implementation Quality**:
- ✅ Comprehensive test coverage for all new functionality
- ✅ Validation tests for consistency calculation with known data
- ✅ Integration tests for complete data persistence pipeline
- ✅ Performance tests ensuring minimal overhead from enhanced logging
- ✅ Backward compatibility tests with existing result file formats

**Analytical Notebook Readiness**:
- ✅ Variant dimensions retrievable from all result files
- ✅ Per-input consistency data available for distribution analysis
- ✅ Prediction distributions enable rich visualization of model uncertainty
- ✅ Complete audit trail supports debugging and error analysis
- ✅ Dimensional consistency analysis supports systematic robustness evaluation

### **🎯 PHASE 2 SUCCESS CRITERIA (AFTER REFACTOR)**
**Context Selection & Validation**:
- ✅ Select diverse context examples using Length + TF-IDF from train split only
- ✅ Mandatory contamination validation: zero overlap between context examples and test samples
- ✅ Generate context selection audit trail with TF-IDF analysis and selection reasoning
- ✅ Store complete context example metadata (length category, dissimilarity scores)

**Model Selection & Enhancement**:
- ✅ Identify best-performing model-prompt combination using corrected consistency calculation
- ✅ Generate prefix and suffix context variants based on best baseline combination
- ✅ Store enhanced prompt templates with context positioning metadata

**Evaluation & Data Persistence**:
- ✅ Evaluate enhanced 18-variant system (16 baseline + 2 context-enhanced)
- ✅ Store complete execution data: formatted prompts, raw responses, context examples used
- ✅ Generate Phase 2 detailed execution data with per-input analysis
- ✅ Create baseline vs context-enhanced comparison analysis
- ✅ Store research insights: context effectiveness, cost-benefit analysis, improvement patterns

**Data Integrity & Traceability**:
- ✅ Complete audit trail of context selection process and rationale
- ✅ Reference links to Phase 1.5 baseline results for traceability
- ✅ Validation logs proving no data contamination between splits
- ✅ Token usage analysis and cost implications of context enhancement

---

## 💡 **FOR FUTURE AI ASSISTANTS**

**Current State**: Phase 1, Phase 1.5, and Phase 2 are ALL COMPLETELY IMPLEMENTED and TESTED. All major design decisions are APPROVED. The complete baseline + context enhancement system is production-ready.

**✅ COMPLETED & FULLY TESTED**:
- ✅ Complete prompt system (16 variants + flexible grouping)
- ✅ Data loading with balanced sampling and contamination prevention
- ✅ Custom evaluation metrics (corrected polarity-weighted accuracy + consistency)
- ✅ Flexible model architecture with runtime CLI selection (`--models gpt-4.1,gpt-4o-mini`)
- ✅ Enhanced configuration management system supporting easy model switching
- ✅ Complete data persistence with audit trails (comprehensive output formats)
- ✅ Unicode-safe logging with Windows emoji compatibility
- ✅ Context enhancement system with length + TF-IDF diverse selection
- ✅ Multi-model best variant selection (per model, not single overall best)
- ✅ Prefix and suffix context integration with complete template generation
- ✅ Comprehensive testing (208 tests with 100% pass rate)

**✅ CLEANUP COMPLETED**:
- ✅ Removed "_enhanced" suffixes from file names for professional final codebase
- ✅ Renamed "test_phase_1_5_" prefixes to standard test naming conventions
- ✅ Consolidated redundant configuration files
- ✅ Standardized logging configuration across all scripts
- ✅ Repository is now in clean, production-ready state

**✅ VALIDATION COMPLETED**:
- ✅ Final consistency calculation logic verified and working correctly
- ✅ Enhanced data persistence strategy validated with 3-file output format
- ✅ Model selection CLI design confirmed working (`--models gpt-4o-mini`)
- ✅ All functionality integrated seamlessly into main codebase

**🎉 PHASE 2 COMPLETE**: Context enhancement system operational with zero contamination validation, multi-model best selection, and comprehensive data persistence

**🤝 STILL NEEDS CONSULTATION**: PEFT configuration, analysis visualization, final report structure

**Key Point**: All phases COMPLETE with excellent testing results (208/208 tests passing, full end-to-end validation, zero data contamination confirmed). Complete baseline + context enhancement system ready for production research experiments.

---

## 🔧 **DEVELOPMENT COMMANDS**

### **Environment**
```bash
poetry install          # Install dependencies
poetry shell           # Activate environment
poetry run pytest      # Run tests
poetry run pytest --cov # With coverage
```

### **Configuration & Setup**
```bash
# Verify current configuration (after Phase 1.5 refactor)
poetry run python -c "from deep_learning_final_assignment.core.config import load_experiment_config; config = load_experiment_config(); print('Enabled Models:', config.get_enabled_models())"

# Note: Ollama setup script (scripts/setup_ollama.py) needs refactoring for flexible model support
```

### **Execution (All Phases Complete - Production Ready)**
```bash
# PRIMARY RESEARCH FOCUS: GPT-4.1 vs GPT-4o-mini comparison with Phase 2 context enhancement
poetry run python scripts/run_baseline_async.py --models gpt-4.1,gpt-4o-mini --n-samples 50 --run-phase2

# Fast testing with Phase 2 context enhancement
poetry run python scripts/run_baseline_async.py --models gpt-4o-mini --n-samples 10 --run-phase2

# Baseline only (no context enhancement)
poetry run python scripts/run_baseline_async.py --models gpt-4o-mini --n-samples 10

# Quick validation (2 samples with Phase 2, confirmed working)
poetry run python scripts/run_baseline_async.py --models gpt-4o-mini --n-samples 2 --run-phase2 --log-level INFO

# Alternative model combination (flexible architecture)
poetry run python scripts/run_baseline_async.py --models qwen3:4b --run-phase2

# Run analysis notebook (when results available)
jupyter notebook notebooks/analysis.ipynb
```

### **Dependencies (Already Installed)**
- Core ML/NLP: transformers, datasets, torch, scikit-learn, sentence-transformers
- API Integration: openai, ollama
- Evaluation: rouge-score, pandas, python-dotenv, jupyter
- PEFT: peft, accelerate
- Testing: pytest, pytest-cov, pytest-mock, pytest-asyncio

### **Target Model Configuration (After Phase 1.5)**
- **Primary Focus**: `gpt-4.1` vs `gpt-4o-mini` comparison
- **Flexible Architecture**: Easy switching to other models via configuration
- **Configuration**: Enhanced JSON-based parameterization in `config/experiment_config.json`
- **Test Config**: Separate test configuration in `config/test_config.json`
- **Extensibility**: Support for Ollama models (`qwen3:4b`, etc.) and future providers

---

**REMEMBER**: 
- ✅ **Phase 1 Core Logic COMPLETE**: All 16 prompt variants implemented, tested, and ready for execution
- 🔄 **Phase 1 Model Architecture NEEDS REFACTOR**: Currently tied to specific models, needs flexible switching
- ✅ **Testing**: 154 comprehensive tests with 100% pass rate (core logic)
- 🔄 **Phase 1.5 CRITICAL**: Model architecture decoupling + consistency calculation refactor + data persistence
- 🚀 **Primary Research Focus**: GPT-4.1 vs GPT-4o-mini comparison (easy switching via config)
- 🎯 **Flexible Architecture Goal**: Support for any model combination (OpenAI, Ollama, future providers)

**Next Steps**: Implement Phase 1.5 refactor FIRST (model flexibility + consistency + persistence), THEN proceed with Phase 2 (Context Enhancement) using the primary research focus models. 

---

---

## 🔄 **PHASE 1.5 REFACTORING STRATEGY (DETAILED)**

### **Critical Issues Being Fixed**

#### **Issue 1: Incorrect Consistency Calculation**
**Current Problem**: Each individual combination gets `consistency_score = 1.0`, then all variants of a model get the same cross-variant consistency score applied retroactively.

**Correct Logic**: 
1. **Per-Input Consistency**: For each test input, calculate how many variants agree on the prediction
2. **Model Consistency**: Average the per-input consistency scores across all test inputs
3. **Weighted Index**: Use model-level consistency for all variants of that model

#### **Issue 2: Missing Data Persistence**
**Current Problem**: Only final predictions and evaluation metrics are saved. Missing:
- Formatted prompts sent to models
- Raw model responses received
- Variant dimensional information
- Per-input consistency analysis
- Prediction distributions

**Solution**: Complete audit trail with multiple output file formats

#### **Issue 3: Inflexible Model Selection**
**Current Problem**: Both OpenAI and Ollama models are always initialized and run.

**Solution**: Configuration-driven model selection with CLI overrides for speed testing.

### **Enhanced Data Structures**

#### **InputConsistencyData**
```python
@dataclass
class InputConsistencyData:
    input_id: int
    input_text: str
    true_label: str
    variant_predictions: Dict[str, str]           # variant_id -> prediction
    variant_dimensions: Dict[str, Dict[str, str]] # variant_id -> dimensions
    prediction_distribution: Dict[str, float]     # label -> percentage
    consistency_score: float                      # max(distribution.values())
    majority_label: str                          # most frequent prediction
    formatted_prompts: Dict[str, str]            # variant_id -> formatted_prompt
    raw_responses: Dict[str, ModelResponse]      # variant_id -> raw_response
```

#### **ModelConsistencyData**
```python
@dataclass
class ModelConsistencyData:
    model_name: str
    overall_consistency: float                    # mean(per_input_consistency)
    per_input_consistency: List[float]           # consistency score per input
    input_consistency_data: List[InputConsistencyData]
    dimensional_consistency: Dict[str, float]    # consistency by dimension
    total_inputs: int
    total_variants: int
```

### **Enhanced File Output Strategy**

#### **1. Detailed Execution Data** (`detailed_execution_data_TIMESTAMP.json`)
```json
{
  "experiment_metadata": {
    "timestamp": "2025-01-25T14:30:00Z",
    "models_evaluated": ["gpt-4.1", "gpt-4o-mini"],
    "total_combinations": 32,
    "total_inputs": 50,
    "configuration": {...}
  },
  "model_name": {
    "per_input_analysis": [
      {
        "input_id": 0,
        "input_text": "Great movie!",
        "true_label": "Positive",
        "variant_predictions": {
          "v1": "Positive",
          "v2": "Very Positive",
          "v3": "Positive",
          // ... all 16 variants
        },
        "variant_dimensions": {
          "v1": {"formality": "formal", "phrasing": "imperative", "order": "task_first", "synonyms": "set_a"},
          "v2": {"formality": "formal", "phrasing": "imperative", "order": "task_first", "synonyms": "set_b"},
          // ... all 16 variants
        },
        "prediction_distribution": {
          "Very Negative": 0.0,
          "Negative": 0.0,
          "Neutral": 0.0,
          "Positive": 0.75,      // 12/16 variants
          "Very Positive": 0.25  // 4/16 variants
        },
        "consistency_score": 0.75,
        "majority_label": "Positive",
        "formatted_prompts": {
          "v1": "Analyze the sentiment of the following text...",
          // ... all 16 formatted prompts
        },
        "raw_responses": {
          "v1": {
            "content": "{\"sentiment\": \"Positive\"}",
            "model_name": "gpt-4.1",
            "metadata": {...}
          },
          // ... all 16 raw responses
        }
      }
      // ... for all 50 inputs
    ],
    "model_summary": {
      "overall_consistency": 0.68,
      "dimensional_consistency": {
        "formality": 0.72,
        "phrasing": 0.65,
        "order": 0.70,
        "synonyms": 0.66
      }
    }
  }
}
```

#### **2. Enhanced Detailed Results** (`baseline_results_detailed_TIMESTAMP.json`)
```json
{
  "model_name": {
    "variant_id": {
      "model_name": "gpt-4.1",
      "prompt_variant_id": "v1",
      "variant_dimensions": {"formality": "formal", "phrasing": "imperative", ...},
      "variant_name": "Formal + Imperative + Task-first + Synonym A",
      "custom_accuracy": 0.90,
      "consistency_score": 0.68,        // Model-level consistency
      "weighted_index": 0.834,          // 0.7*0.90 + 0.3*0.68
      "error_breakdown": {...},
      "metadata": {
        "dimensional_consistency": {...},
        "overall_consistency": 0.68,
        "total_samples": 50
      },
      "predictions": ["Positive", "Negative", ...],
      "true_labels": ["Positive", "Neutral", ...],
      "execution_metadata": {
        "avg_response_time": 1.2,
        "total_tokens_used": 1500,
        "error_count": 0
      }
    }
    // ... all 16 variants
  }
}
```

#### **3. Summary Results** (`baseline_results_summary_TIMESTAMP.json`)
```json
{
  "model_name": {
    "performance_summary": {
      "total_combinations": 16,
      "accuracy_stats": {...},
      "consistency_stats": {
        "mean": 0.68,
        "std": 0.0,      // Same for all variants
        "min": 0.68,
        "max": 0.68
      },
      "weighted_index_stats": {...}
    },
    "best_combination": {
      "variant_id": "v11",
      "custom_accuracy": 0.92,
      "consistency_score": 0.68,
      "weighted_index": 0.848
    },
    "dimensional_analysis": {
      "formality": {"formal": 0.85, "casual": 0.87},
      "phrasing": {"imperative": 0.86, "question": 0.86},
      "order": {"task_first": 0.85, "text_first": 0.87},
      "synonyms": {"set_a": 0.86, "set_b": 0.86}
    },
    "total_variants_tested": 16
  }
}
```

### **Flexible Model Selection Implementation**

#### **Enhanced Configuration**
```json
{
  "models": {
    "enabled_models": ["gpt-4o-mini"],     // NEW: Control which models to run
    "openai": {
      "model_options": {
        "gpt-4.1": {
          "temperature": 0.1,
          "max_tokens": 150,
          "json_mode": true
        },
        "gpt-4o-mini": {
          "temperature": 0.1,
          "max_tokens": 150,
          "json_mode": true
        }
      }
    },
    "ollama": {
      "model_options": {
        "qwen3:4b": {
          "temperature": 0.1,
          "max_tokens": 150,
          "host": "http://localhost:11434"
        }
      }
    }
  }
}
```

#### **CLI Model Selection**
```bash
# Primary research focus: GPT-4.1 vs GPT-4o-mini
python scripts/run_baseline_async.py --models gpt-4.1,gpt-4o-mini

# Fast testing with only GPT-4o-mini
python scripts/run_baseline_async.py --models gpt-4o-mini

# Alternative model (flexible architecture)
python scripts/run_baseline_async.py --models qwen3:4b
```

### **Implementation Sequence**

1. **Enhanced Data Structures**: Create new dataclasses for comprehensive data storage
2. **Corrected Consistency Logic**: Fix per-input → model-level consistency calculation
3. **Enhanced Persistence**: Implement complete audit trail storage
4. **Model Selection**: Add configuration and CLI support for flexible model selection
5. **Delayed Calculations**: Fix weighted index timing and logging
6. **Test Updates**: Comprehensive test coverage for all changes
7. **Backward Compatibility**: Ensure existing result formats still work

### **Validation Strategy**

#### **Consistency Calculation Validation**
```python
# Test with known data
variant_predictions = {
    "v1": ["Positive", "Negative", "Positive"],
    "v2": ["Positive", "Positive", "Positive"], 
    "v3": ["Very Positive", "Negative", "Positive"]
}

# Expected per-input consistency: [0.33, 0.33, 1.0]
# Expected model consistency: (0.33 + 0.33 + 1.0) / 3 = 0.55
```

#### **Data Persistence Validation**
- Verify all formatted prompts are stored
- Verify all raw responses are captured
- Verify variant dimensions are preserved
- Verify prediction distributions sum to 1.0
- Verify no data loss during serialization

#### **Model Selection Validation**
- Test with `enabled_models: ["gpt-4o-mini"]` runs only GPT-4o-mini
- Test with `enabled_models: ["gpt-4.1", "gpt-4o-mini"]` runs both models
- Test with `enabled_models: ["qwen3:4b"]` runs only Ollama model (flexible architecture)
- Test CLI overrides work correctly
- Test backward compatibility with existing configs

---

## 📊 **ANALYTICAL NOTEBOOK DATA AVAILABILITY (POST PHASE 1.5)**

### **Complete Dimensional Analysis Support**

#### **Variant Dimensions Retrieval**
After Phase 1.5, variant dimensions will be available in multiple formats:

```python
# From detailed results file
with open('baseline_results_detailed_TIMESTAMP.json') as f:
    results = json.load(f)
    
# Access variant dimensions for any combination
gpt4_1_v1_dimensions = results['gpt-4.1']['v1']['variant_dimensions']
# Returns: {"formality": "formal", "phrasing": "imperative", "order": "task_first", "synonyms": "set_a"}

# Access variant name for readable analysis
gpt4_1_v1_name = results['gpt-4.1']['v1']['variant_name'] 
# Returns: "Formal + Imperative + Task-first + Synonym A"
```

#### **Per-Input Consistency Analysis**
```python
# From detailed execution data file
with open('detailed_execution_data_TIMESTAMP.json') as f:
    execution_data = json.load(f)

# Analyze consistency patterns per input
for input_data in execution_data['gpt-4.1']['per_input_analysis']:
    input_text = input_data['input_text']
    consistency = input_data['consistency_score']
    distribution = input_data['prediction_distribution']
    
    # Rich analysis possible:
    # - Which inputs have low consistency?
    # - What prediction patterns emerge?
    # - How do dimensions affect specific inputs?
```

#### **Dimensional Consistency Patterns**
```python
# Compare consistency across dimensions
dimensional_consistency = execution_data['gpt-4.1']['model_summary']['dimensional_consistency']

# Example output:
# {
#   "formality": 0.72,    # Formal vs Casual variants
#   "phrasing": 0.65,     # Imperative vs Question variants  
#   "order": 0.70,        # Task-first vs Text-first variants
#   "synonyms": 0.66      # Set A vs Set B variants
# }

# Identify which dimensions cause most inconsistency
least_consistent_dimension = min(dimensional_consistency, key=dimensional_consistency.get)
```

### **Prediction Distribution Analysis**

#### **Model Uncertainty Visualization**
```python
# For each input, analyze prediction uncertainty
for input_data in execution_data['gpt-4.1']['per_input_analysis']:
    distribution = input_data['prediction_distribution']
    
    # Visualize uncertainty patterns:
    # - High entropy: Model uncertain (e.g., {"Positive": 0.4, "Very Positive": 0.4, "Neutral": 0.2})
    # - Low entropy: Model confident (e.g., {"Positive": 0.9, "Very Positive": 0.1})
    # - Polarity confusion: Cross-boundary errors (e.g., {"Positive": 0.5, "Negative": 0.3})
```

#### **Systematic Error Analysis**
```python
# Identify systematic biases across variants
for input_data in execution_data['gpt-4.1']['per_input_analysis']:
    true_label = input_data['true_label']
    majority_prediction = input_data['majority_label']
    
    if true_label != majority_prediction:
        # Analyze which variants contributed to the error
        variant_predictions = input_data['variant_predictions']
        variant_dimensions = input_data['variant_dimensions']
        
        # Group errors by dimensional patterns
        error_patterns = {}
        for variant_id, prediction in variant_predictions.items():
            if prediction != true_label:
                dimensions = variant_dimensions[variant_id]
                # Track which dimensional combinations lead to errors
```

### **Cross-Model Comparison Support**

#### **Model Robustness Comparison**
```python
# Primary research focus: Compare GPT-4.1 vs GPT-4o-mini consistency
gpt4_1_consistency = execution_data['gpt-4.1']['model_summary']['overall_consistency']
gpt4_mini_consistency = execution_data['gpt-4o-mini']['model_summary']['overall_consistency']

# Compare dimensional sensitivities
gpt4_1_dim_consistency = execution_data['gpt-4.1']['model_summary']['dimensional_consistency']
gpt4_mini_dim_consistency = execution_data['gpt-4o-mini']['model_summary']['dimensional_consistency']

# Identify which model is more robust to which dimensions
for dimension in ['formality', 'phrasing', 'order', 'synonyms']:
    gpt4_1_score = gpt4_1_dim_consistency[dimension]
    gpt4_mini_score = gpt4_mini_dim_consistency[dimension]
    more_robust = "GPT-4.1" if gpt4_1_score > gpt4_mini_score else "GPT-4o-mini"
    print(f"{dimension}: {more_robust} more robust ({gpt4_1_score:.3f} vs {gpt4_mini_score:.3f})")
```

### **Audit Trail and Debugging Support**

#### **Complete Execution Reconstruction**
```python
# For any combination, reconstruct the complete execution
model_name = "gpt-4.1"
variant_id = "v1"
input_id = 0

# Get the exact prompt sent to the model
formatted_prompt = execution_data[model_name]['per_input_analysis'][input_id]['formatted_prompts'][variant_id]

# Get the raw response received
raw_response = execution_data[model_name]['per_input_analysis'][input_id]['raw_responses'][variant_id]

# Get the parsed prediction
prediction = execution_data[model_name]['per_input_analysis'][input_id]['variant_predictions'][variant_id]

# Full traceability for debugging and analysis
```

#### **Error Pattern Investigation**
```python
# Identify inputs where models disagree most
high_disagreement_inputs = []
for input_data in execution_data['gpt-4.1']['per_input_analysis']:
    if input_data['consistency_score'] < 0.5:  # High disagreement
        high_disagreement_inputs.append({
            'input_text': input_data['input_text'],
            'true_label': input_data['true_label'],
            'predictions': input_data['variant_predictions'],
            'dimensions': input_data['variant_dimensions']
        })

# Analyze patterns in high-disagreement cases
```

### **Research Insights Enabled**

#### **Prompt Engineering Insights**
- **Formality Impact**: Compare formal vs casual variant performance
- **Phrasing Sensitivity**: Analyze imperative vs question effectiveness  
- **Order Effects**: Measure task-first vs text-first impact
- **Synonym Robustness**: Evaluate technical vs common terminology

#### **Model Characterization**
- **Consistency Profiles**: Each model's robustness signature across dimensions
- **Error Patterns**: Systematic biases and failure modes
- **Uncertainty Quantification**: Prediction confidence and distribution patterns
- **Dimensional Interactions**: How multiple prompt dimensions interact

#### **Practical Applications**
- **Prompt Optimization**: Identify most robust prompt formulations
- **Model Selection**: Choose models based on consistency requirements
- **Risk Assessment**: Quantify deployment risks from prompt sensitivity
- **Quality Assurance**: Systematic testing framework for prompt variations
```

## 🔄 **PHASE 2 DATA PERSISTENCE STRATEGY (CONTEXT ENHANCEMENT)**

### **Enhanced File Output for Context-Enhanced Evaluation**

Phase 2 must maintain the same comprehensive data persistence standards as Phase 1.5, with additional context-specific data.

#### **1. Context Selection Audit Trail** (`context_selection_audit_TIMESTAMP.json`)
```json
{
  "context_selection_metadata": {
    "timestamp": "2025-01-25T15:30:00Z",
    "selection_strategy": "length_diversity_plus_tfidf",
    "data_split_used": "train",
    "total_train_samples": 8544,
    "selected_examples_per_label": 3,
    "total_context_examples": 15,
    "validation_split_used": "validation",
    "test_samples": 50,
    "contamination_check": "PASSED"
  },
  "selected_context_examples": {
    "Very Negative": [
      {
        "text": "This movie is absolutely terrible...",
        "label": "Very Negative",
        "length_category": "short",
        "tfidf_dissimilarity_score": 0.85,
        "selection_reason": "high_dissimilarity_short_text"
      },
      {
        "text": "I cannot express how disappointed I am...",
        "label": "Very Negative", 
        "length_category": "medium",
        "tfidf_dissimilarity_score": 0.79,
        "selection_reason": "high_dissimilarity_medium_text"
      },
      {
        "text": "The entire experience was a complete waste...",
        "label": "Very Negative",
        "length_category": "long", 
        "tfidf_dissimilarity_score": 0.82,
        "selection_reason": "high_dissimilarity_long_text"
      }
    ],
    // ... for all 5 labels
  },
  "contamination_validation": {
    "overlap_check_method": "exact_text_match",
    "overlapping_examples": 0,
    "validation_status": "NO_CONTAMINATION_DETECTED",
    "test_sample_hashes": ["hash1", "hash2", ...],
    "context_example_hashes": ["hash_a", "hash_b", ...]
  },
  "tfidf_analysis": {
    "vectorizer_params": {...},
    "vocabulary_size": 5000,
    "dissimilarity_matrix": {...},
    "selection_algorithm": "greedy_max_dissimilarity"
  }
}
```

#### **2. Enhanced Prompt Variants** (`context_enhanced_prompts_TIMESTAMP.json`)
```json
{
  "best_baseline_combination": {
    "model_name": "gpt-4.1",
    "variant_id": "v11",
    "weighted_index": 0.848,
    "original_template": "Check out this text above and figure out...",
    "variant_dimensions": {"formality": "casual", "phrasing": "imperative", "order": "text_first", "synonyms": "set_a"}
  },
  "context_enhanced_variants": {
    "prefix_context": {
      "variant_id": "v17_prefix",
      "template_structure": "[CONTEXT_EXAMPLES] + [ORIGINAL_PROMPT]",
      "full_template": "Here are examples of sentiment classification:\n\nVery Negative: 'This movie is absolutely terrible...'\n...\n\nNow check out this text above and figure out the sentiment...",
      "context_position": "prefix",
      "base_variant": "v11"
    },
    "suffix_context": {
      "variant_id": "v18_suffix", 
      "template_structure": "[ORIGINAL_PROMPT] + [CONTEXT_EXAMPLES]",
      "full_template": "Check out this text above and figure out the sentiment...\n\nHere are examples:\nVery Negative: 'This movie is absolutely terrible...'\n...",
      "context_position": "suffix",
      "base_variant": "v11"
    }
  },
  "context_formatting": {
    "example_format": "{label}: '{text}'",
    "separator": "\n",
    "context_introduction": "Here are examples of sentiment classification:",
    "total_context_length_prefix": 450,
    "total_context_length_suffix": 445
  }
}
```

#### **3. Phase 2 Detailed Execution Data** (`phase2_execution_data_TIMESTAMP.json`)
```json
{
  "experiment_metadata": {
    "phase": "2_context_enhancement",
    "timestamp": "2025-01-25T15:45:00Z",
    "base_model": "gpt-4.1",
    "base_variant": "v11",
    "total_variants_tested": 18,
    "baseline_variants": 16,
    "context_enhanced_variants": 2,
    "test_samples": 50,
    "context_examples_used": 15
  },
  "baseline_results_reference": {
    "file_path": "baseline_results_detailed_20250125_1430.json",
    "best_combination": {
      "model_name": "gpt-4.1",
      "variant_id": "v11", 
      "weighted_index": 0.848
    }
  },
  "context_enhanced_results": {
    "v17_prefix": {
      "per_input_analysis": [
        {
          "input_id": 0,
          "input_text": "Great movie!",
          "true_label": "Positive",
          "prediction": "Positive",
          "formatted_prompt": "Here are examples of sentiment classification:\n\nVery Negative: 'This movie is absolutely terrible...'\n...\n\nText: Great movie!\n\nCheck out this text above and figure out the sentiment...",
          "raw_response": {
            "content": "{\"sentiment\": \"Positive\"}",
            "model_name": "gpt-4.1",
            "metadata": {
              "total_tokens": 245,
              "context_tokens": 195,
              "response_time": 1.4
            }
          },
          "context_examples_used": [
            {"label": "Very Negative", "text": "This movie is absolutely terrible..."},
            // ... all 15 context examples
          ]
        }
        // ... for all 50 inputs
      ],
      "variant_summary": {
        "custom_accuracy": 0.94,
        "baseline_comparison": {
          "baseline_accuracy": 0.92,
          "improvement": 0.02,
          "improvement_percentage": 2.17
        },
        "error_analysis": {
          "errors_fixed_by_context": 1,
          "new_errors_introduced": 0,
          "net_improvement": 1
        }
      }
    },
    "v18_suffix": {
      // Similar structure for suffix variant
    }
  },
  "comparative_analysis": {
    "baseline_vs_context": {
      "baseline_best_accuracy": 0.92,
      "prefix_context_accuracy": 0.94,
      "suffix_context_accuracy": 0.93,
      "best_overall": "v17_prefix",
      "context_effectiveness": {
        "prefix_improvement": 0.02,
        "suffix_improvement": 0.01,
        "statistical_significance": "pending_analysis"
      }
    },
    "per_label_analysis": {
      "Very Negative": {
        "baseline_accuracy": 0.90,
        "prefix_accuracy": 0.95,
        "suffix_accuracy": 0.92
      },
      // ... for all labels
    }
  }
}
```

#### **4. Phase 2 Summary Results** (`phase2_results_summary_TIMESTAMP.json`)
```json
{
  "experiment_summary": {
    "phase": "2_context_enhancement",
    "baseline_reference": "baseline_results_summary_20250125_1430.json",
    "best_baseline": {
      "model_name": "gpt-4.1",
      "variant_id": "v11",
      "accuracy": 0.92,
      "weighted_index": 0.848
    },
    "context_enhanced_results": {
      "v17_prefix": {
        "accuracy": 0.94,
        "improvement_over_baseline": 0.02,
        "context_position": "prefix",
        "total_tokens_avg": 245
      },
      "v18_suffix": {
        "accuracy": 0.93, 
        "improvement_over_baseline": 0.01,
        "context_position": "suffix",
        "total_tokens_avg": 240
      }
    },
    "best_overall": {
      "variant_id": "v17_prefix",
      "accuracy": 0.94,
      "context_position": "prefix",
      "improvement": 0.02,
      "tokens_cost_increase": "25%"
    }
  },
  "research_insights": {
    "context_effectiveness": "prefix_superior",
    "improvement_magnitude": "modest_but_consistent", 
    "cost_benefit_analysis": {
      "accuracy_gain": 0.02,
      "token_cost_increase": 0.25,
      "cost_per_accuracy_point": "12.5x_tokens"
    },
    "robustness_impact": {
      "baseline_consistency": 0.68,
      "context_enhanced_consistency": "to_be_calculated_in_phase_3"
    }
  }
}
```

### **Data Separation Validation Requirements**

#### **Mandatory Contamination Checks**
```python
# REQUIRED: Automatic validation before Phase 2 execution
def validate_no_contamination(context_examples: List[str], test_samples: List[str]) -> bool:
    """
    Ensure zero overlap between context examples (from train) and test samples (from validation).
    Returns True if no contamination detected, raises Exception if contamination found.
    """
    context_hashes = {hash(example.strip().lower()) for example in context_examples}
    test_hashes = {hash(sample.strip().lower()) for sample in test_samples}
    
    overlap = context_hashes.intersection(test_hashes)
    if overlap:
        raise ValueError(f"CONTAMINATION DETECTED: {len(overlap)} overlapping examples found")
    
    return True

# REQUIRED: Log validation results
validation_log = {
    "contamination_check": "PASSED" if validate_no_contamination(...) else "FAILED",
    "context_source": "train_split", 
    "test_source": "validation_split",
    "validation_timestamp": datetime.now().isoformat()
}
```

#### **Data Split Enforcement**
```python
# REQUIRED: Explicit split enforcement
class Phase2DataLoader:
    def get_context_examples(self) -> List[ContextExample]:
        """MUST use train split only"""
        return self.sst5_loader.get_diverse_examples(
            split="train",  # CRITICAL: Never use validation/test
            n_per_label=3
        )
    
    def get_test_samples(self) -> List[TestSample]:  
        """MUST use validation split only"""
        return self.sst5_loader.get_sample_subset(
            split="validation",  # CRITICAL: Never use train
            n_samples=50,
            balanced=True
        )
```

---