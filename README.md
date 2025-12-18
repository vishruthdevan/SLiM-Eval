# SLiM-Eval

**Small Language Model Evaluation Framework** — Comprehensive benchmarking for quantized LLMs across performance, energy, and accuracy metrics.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

SLiM-Eval is a unified framework for evaluating Large Language Models (LLMs) with different quantization strategies. It measures:

- **Performance**: Latency (TTFT, TPOT, E2E) and GPU memory usage
- **Energy**: Power consumption and energy efficiency
- **Accuracy**: Model quality on standard benchmarks (MMLU, GSM8K, HellaSwag)

### Supported Quantization Methods

| Precision | Method | Description |
|-----------|--------|-------------|
| `fp16` | Baseline | Half-precision floating point (no quantization) |
| `int8` | GPTQ | 8-bit weights and activations (W8A8) |
| `int4` | GPTQ | 4-bit weights, 16-bit activations (W4A16) |

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- CUDA 11.8+ and cuDNN

### Setup

```bash
# Clone the repository
git clone https://github.com/vishruthdevan/SLiM-Eval.git
cd SLiM-Eval

# Install dependencies
pip install -e .

# Optional: Install Jupyter for notebooks
pip install -e ".[dev]"
```

## Quick Start

### Environment Setup

Set up the required environment variables before running benchmarks:

```bash
# Required for accessing gated models (e.g., Llama, Gemma)
export HF_TOKEN=your_huggingface_token

# Optional: Enable Weights & Biases logging
export WANDB_API_KEY=your_wandb_api_key
```

### Basic Usage

Run a complete benchmark suite on a model:

```bash
slim-eval run \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --precision fp16 \
  --output-dir outputs
```

This will run the full benchmark suite with:

- 10 warmup runs + 500 measured runs for stable statistics
- Batch size of 8 for improved throughput
- 256 token generation
- Full accuracy evaluation (MMLU, GSM8K, HellaSwag) with 5-shot
- 200 energy sample runs for stable power estimates
- Weights & Biases logging enabled by default

To run multiple precisions, execute separate commands for each:

```bash
slim-eval run --models "meta-llama/Llama-3.2-3B-Instruct" --precision fp16 --max-model-len 8192
slim-eval run --models "meta-llama/Llama-3.2-3B-Instruct" --precision int8 --max-model-len 8192
slim-eval run --models "meta-llama/Llama-3.2-3B-Instruct" --precision int4 --max-model-len 8192
```

### Performance-Only Benchmark

Quick latency and memory profiling (reduced runs for faster results):

```bash
slim-eval run \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --precision fp16 \
  --tasks performance \
  --num-runs 50 \
  --num-warmup 5
```

### Accuracy Evaluation

Run model quality benchmarks (uses 5-shot by default):

```bash
slim-eval run \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --precision fp16 \
  --tasks accuracy \
  --accuracy-tasks "mmlu gsm8k hellaswag"
```

### Analyze Previous Results

Generate visualizations from saved results:

```bash
slim-eval analyze --input-dir outputs --output-dir analysis_results
```

## CLI Reference

### Main Command: `slim-eval run`

#### Model & Precision Options

- `--models`: HuggingFace model IDs or local paths (space-separated for multiple)
- `--precision`: Quantization precision to evaluate
  - Choices: `fp16`, `int8`, `int4`
  - Default: `fp16`

#### Benchmark Tasks

- `--tasks`: Space-separated list of benchmarks to run
  - `performance`: Latency & memory usage
  - `energy`: Power consumption tracking
  - `accuracy`: Model quality metrics
  - Default: `performance accuracy energy` (full suite)

#### Performance Benchmark Options

- `--num-warmup`: Warmup iterations before measurement (default: 10)
- `--num-runs`: Number of measured inference runs (default: 500)
- `--batch-size`: Concurrent requests per iteration (default: 8)
- `--prompt`: Input prompt for latency tests (default: "Explain one interesting fact about large language models.")
- `--max-new-tokens`: Tokens to generate per request (default: 256)

#### Energy Benchmark Options

- `--energy-sample-runs`: Number of energy-tracked requests (default: 200)

#### Accuracy Benchmark Options

- `--accuracy-tasks`: Space-separated lm-eval tasks to run (default: `mmlu gsm8k hellaswag`)
- `--num-fewshot`: Few-shot examples (default: 5)
- `--accuracy-limit`: Limit examples per task for quick testing (default: None - run full benchmark)
- `--accuracy-batch-size`: Global batch size (default: 32)
- `--accuracy-batch-size-{task}`: Per-task batch size overrides

#### vLLM Configuration

- `--gpu-memory-utilization`: GPU memory fraction for vLLM (default: 0.8)
- `--max-model-len`: Maximum context window for inference (default: 8192)

#### GPU Selection

- `--gpu-index`: Select NVIDIA GPU index to use, 0-based (default: 0)

#### Weights & Biases Integration

- `--wandb-enabled`: Enable Weights & Biases logging (default: True)
- `--wandb-project`: W&B project name (default: `slim-eval`)
- `--wandb-api-key`: W&B API key (or set `WANDB_API_KEY` env var)
- `--wandb-run-name`: W&B run name (leave empty for auto-generation)

#### Quantization Options

- `--calibration-dataset`: Dataset for calibration (default: `HuggingFaceH4/ultrachat_200k`)
- `--calibration-split`: Dataset split (default: `train_sft`)
- `--num-calibration-samples`: Calibration samples (default: 512)
- `--max-sequence-length`: Max sequence length for calibration (default: 2048)

#### Output Options

- `--output-dir`: Results directory (default: `outputs`)
- `--quantized-models-dir`: Pre-quantized model cache (default: `quantized-models`)

### Analysis Command: `slim-eval analyze`

```bash
slim-eval analyze \
  --input-dir outputs \
  --output-dir analysis_results \
  --accuracy-tasks mmlu gsm8k hellaswag
```

#### Analysis Options

- `--input-dir`: Directory containing benchmark results to analyze (default: `outputs`)
- `--output-dir`: Directory to write analysis results (plots, CSVs, etc.) (default: `outputs`)
- `--accuracy-tasks`: Accuracy tasks to include in analysis (default: `mmlu gsm8k hellaswag`)
- `--gpu-index`: Select NVIDIA GPU index to use (default: 0)

## Architecture

```text
slim_eval/
├── cli.py                  # Command-line interface
├── evaluator.py            # Main orchestrator
├── quantization.py         # Quantization management
├── analysis.py             # Results visualization
├── utils.py                # Utilities (caching, model info)
└── benchmarks/
    ├── base.py             # Base benchmark class
    ├── performance.py      # Latency & memory tracking
    ├── energy.py           # Power consumption monitoring
    └── accuracy.py         # lm-eval integration
```

## Output Files

After running benchmarks, the output directory contains:

```text
outputs/
└── {model_name}/
    └── {model_name}_{precision}/
        ├── energy.json           # Energy metrics
        ├── gsm8k.json            # GSM8K accuracy results
        ├── hellaswag.json        # HellaSwag accuracy results
        ├── mmlu.json             # MMLU accuracy results
        └── performance.json      # Latency & memory metrics
```

After running analysis:

```text
analysis_results/
├── complete_results.json         # Combined metrics (JSON)
├── executive_summary.txt         # Human-readable summary
├── quantization_impact.csv       # Quantization comparison
├── results_table.csv             # Combined metrics table
├── results_table.tex             # LaTeX table
├── summary_statistics.csv        # Statistical summary
└── plots/
    ├── latency_comparison.png    # Latency visualizations
    ├── memory_comparison.png     # Memory usage charts
    ├── energy_comparison.png     # Energy efficiency plots
    └── accuracy_comparison.png   # Model quality comparison
```

## Quantized Model Storage

When running benchmarks with `int8` or `int4` precision, SLiM-Eval automatically quantizes models and caches them for future use:

```text
quantized-models/
└── {model_name}_{precision}/
    ├── config.json
    ├── model.safetensors (or model-*.safetensors for sharded models)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

- **Location**: Controlled by `--quantized-models-dir` (default: `quantized-models`)
- **Reuse**: If a quantized model already exists, it will be loaded directly without re-quantization
- **Storage**: Quantized models are typically 2-4x smaller than fp16 models

To force re-quantization, delete the corresponding directory in `quantized-models/`.

## Key Metrics

### Performance Metrics

- **TTFT** (Time to First Token): Initial response latency
- **TPOT** (Time Per Output Token): Per-token generation speed
- **E2E Latency**: Total end-to-end time
- **Throughput**: Tokens generated per second
- **GPU Memory**: Peak memory usage during inference

### Energy Metrics

- **Power Draw**: GPU power consumption (watts)
- **Total Energy**: Energy used per request (joules)
- **Tokens per Joule**: Energy efficiency metric

### Accuracy Metrics

- **MMLU**: Multitask Language Understanding (0-100%)
- **GSM8K**: Grade School Math (exact match %)
- **HellaSwag**: Commonsense reasoning (normalized accuracy %)

## Environment Variables

- `HF_TOKEN`: HuggingFace API token for accessing gated models
- `WANDB_API_KEY`: Weights & Biases API key for logging

## Parameter Guide

### `max_sequence_length` vs `max_model_len`

- **`max_sequence_length`**: Used during **quantization calibration** to limit calibration sample length
- **`max_model_len`**: Used during **inference** to set vLLM's maximum context window

## Examples

### Compare Multiple Models

```bash
# Run each model/precision combination separately
slim-eval run --models "meta-llama/Llama-3.2-1B" --precision fp16 --tasks "performance accuracy"
slim-eval run --models "meta-llama/Llama-3.2-1B" --precision int4 --tasks "performance accuracy"
slim-eval run --models "meta-llama/Llama-3.2-3B" --precision fp16 --tasks "performance accuracy"
slim-eval run --models "meta-llama/Llama-3.2-3B" --precision int4 --tasks "performance accuracy"

# Analyze combined results
slim-eval analyze --input-dir outputs --output-dir multi_model_comparison
```

### Quick Accuracy Check

```bash
slim-eval run \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --precision fp16 \
  --tasks accuracy \
  --accuracy-limit 100 \
  --accuracy-tasks mmlu
```

### Energy-Focused Benchmark

```bash
slim-eval run \
  --models "meta-llama/Llama-3.2-3B-Instruct" \
  --precision fp16 \
  --tasks energy \
  --energy-sample-runs 50
```

### With Weights & Biases Logging

```bash
export WANDB_API_KEY=your_api_key
slim-eval run \
  --models "Qwen/Qwen2.5-3B-Instruct" \
  --precision fp16 \
  --max-model-len 8192 \
  --wandb-enabled \
  --wandb-project slim-eval \
  --wandb-run-name "fp16 Qwen2.5-3B-Instruct" \
  --tasks "energy performance accuracy"
```

## Requirements

Core dependencies (auto-installed):

- PyTorch 2.8.0
- vLLM 0.11.0
- llmcompressor 0.7.1
- transformers 4.55.2
- lm-eval 0.4.9.2
- pandas, matplotlib, seaborn

See `pyproject.toml` for the complete dependency list.

## Citation

If you use SLiM-Eval in your research, please cite:

```bibtex
@software{slim_eval2025,
  author = {Devan, Vishruth and Rajkumar, Kavin Aravindhan},
  title = {SLiM-Eval: Small Language Model Evaluation Framework},
  year = {2025},
  url = {https://github.com/vishruthdevan/SLiM-Eval}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:

- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- [llmcompressor](https://github.com/vllm-project/llm-compressor) for quantization
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for accuracy benchmarks

---

**Maintained by**: [@vishruthdevan](https://github.com/vishruthdevan) and [@KavinAravindhan](https://github.com/KavinAravindhan)
**Issues**: [GitHub Issues](https://github.com/vishruthdevan/SLiM-Eval/issues)
