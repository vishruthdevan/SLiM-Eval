# %% [markdown]
# # SLiM-Eval: Complete Small Language Model Evaluation Framework
# ## Tracking: Latency, Memory, Energy, and Accuracy
#
# This notebook contains:
# 1. Environment setup and verification
# 2. Model quantization using llm-compressor
# 3. **Latency and memory** benchmarking using vLLM
# 4. **Energy consumption** tracking using CodeCarbon/powermetrics
# 5. **Accuracy evaluation** using lm-evaluation-harness
# 6. Results analysis and visualization

# %% [markdown]
# ## Cell 1: Install Required Packages

# %%
# Uncomment and run if packages are not installed
# !pip install torch>=2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install vllm>=0.5.0
# !pip install llm-compressor
# !pip install transformers>=4.40.0
# !pip install accelerate
# !pip install pandas numpy tqdm
# !pip install matplotlib seaborn
# !pip install lm-eval>=0.4.0  # For accuracy evaluation
# !pip install codecarbon  # For energy tracking

# %% [markdown]
# ## Cell 2: Import Libraries and Setup

import gc
import json

# %%
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

# Import codecarbon for energy tracking
from codecarbon import EmissionsTracker

# Import lm-eval for accuracy
from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM as VLLM_LM

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# %% [markdown]
# ## Cell 3: Configuration - Edit This Cell to Customize

# %%
# ============================================================================
# EDITABLE MODEL LIST - Add or remove models as needed
# ============================================================================
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",  # Phi-3 3.8B
    "microsoft/Phi-3.5-mini-instruct",  # Phi-3.5 3.8B
    "google/gemma-2-2b-it",  # Gemma 2 2B
    "meta-llama/Llama-3.2-3B-Instruct",  # Llama 3.2 3B
    "Qwen/Qwen2.5-3B-Instruct",  # Qwen2.5 3B
    "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral 7B
]

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================
# LATENCY & MEMORY BENCHMARKING
NUM_RUNS = 1000  # Number of latency benchmark runs
NUM_WARMUP = 10  # Warmup runs to exclude from metrics
MAX_NEW_TOKENS = 32  # Tokens to generate per inference
BATCH_SIZE = 32  # Batch size for vLLM (higher = faster)
PROMPT = "Explain one interesting fact about large language models."

# ACCURACY EVALUATION
ACCURACY_TASKS = [
    "mmlu",  # MMLU (knowledge)
    "gsm8k",  # GSM8K (math reasoning)
    "hellaswag",  # HellaSwag (commonsense reasoning)
    # "humaneval",       # HumanEval (code) - requires unsafe code execution
]
NUM_FEW_SHOT = 5  # Few-shot examples for accuracy tasks
ACCURACY_LIMIT = None  # Set to small number (e.g., 100) for quick testing

# ENERGY TRACKING
ENABLE_ENERGY_TRACKING = True  # Enable/disable energy monitoring
ENERGY_SAMPLE_RUNS = 100  # Number of runs for energy measurement

# ============================================================================
# QUANTIZATION PRECISIONS TO TEST
# ============================================================================
PRECISIONS = ["fp16", "int8", "int4"]  # Can add "gptq", "awq" later

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_DIR = Path("slim_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "complete_results.csv"
QUANTIZED_MODELS_DIR = Path("quantized_models")
QUANTIZED_MODELS_DIR.mkdir(exist_ok=True)

print(f"Configuration loaded:")
print(f"  Models: {len(MODELS)}")
print(f"  Precisions: {PRECISIONS}")
print(f"  Latency runs: {NUM_RUNS}")
print(f"  Accuracy tasks: {ACCURACY_TASKS}")
print(f"  Energy tracking: {ENABLE_ENERGY_TRACKING}")
print(f"  Output directory: {OUTPUT_DIR}")

# %% [markdown]
# ## Cell 4: Utility Functions


# %%
def clear_cache():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def print_gpu_memory_status():
    """Print current GPU memory status."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | {total:.2f}GB total"
        )


print("Utility functions loaded ✓")
print_gpu_memory_status()

# %% [markdown]
# ## Cell 5: Quantization Configuration

from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

# %%
from llmcompressor.transformers import oneshot

# Quantization configurations
QUANTIZATION_CONFIGS = {
    "int8": {
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True},
                "targets": ["Linear"],
            }
        }
    },
    "int4": {
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 128,
                },
                "targets": ["Linear"],
            }
        }
    },
    "gptq": {
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "group_size": 128,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head"],
    },
}

CALIBRATION_DATASET = "wikitext"
CALIBRATION_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 512

print("Quantization configurations loaded ✓")

# %% [markdown]
# ## Cell 6: Quantization Function (Skip if using pre-quantized models)


# %%
def quantize_model(model_name: str, precision: str, output_dir: Path):
    """Quantize a model using llm-compressor."""
    print(f"\n{'=' * 60}")
    print(f"Quantizing {model_name} to {precision.upper()}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"✓ Already quantized, skipping...")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if precision not in QUANTIZATION_CONFIGS:
            print(f"✗ Unsupported precision: {precision}")
            return

        quant_config = QUANTIZATION_CONFIGS[precision]

        if precision == "gptq":
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"],
            )
        else:
            recipe = QuantizationModifier(**quant_config)

        oneshot(
            model=model_name,
            dataset=CALIBRATION_DATASET,
            dataset_config_name=CALIBRATION_SPLIT,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            recipe=recipe,
            output_dir=str(output_dir),
        )

        print(f"✓ Quantization complete: {output_dir}")

    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()


print("Quantization function loaded ✓")

# %% [markdown]
# ## Cell 7: vLLM Model Setup Function


# %%
def setup_vllm_model(
    model_name: str, precision: str, use_quantized_dir: bool = True
) -> Optional[LLM]:
    """Setup vLLM model with specified precision."""
    clear_cache()

    try:
        print(f"\n{'=' * 60}")
        print(f"Loading {model_name} in {precision.upper()} precision...")
        print(f"{'=' * 60}")

        if precision == "fp16":
            model_path = model_name
            dtype = "float16"
            quantization = None
        else:
            model_short_name = model_name.split("/")[-1]
            quantized_path = QUANTIZED_MODELS_DIR / f"{model_short_name}_{precision}"

            if use_quantized_dir and quantized_path.exists():
                model_path = str(quantized_path)
                print(f"Using pre-quantized model from: {quantized_path}")
            else:
                model_path = model_name
                print(f"Using base model with on-the-fly quantization")

            if precision == "int8":
                dtype = "auto"
                quantization = "int8"
            elif precision == "int4":
                dtype = "auto"
                quantization = "int4"
            elif precision == "gptq":
                dtype = "auto"
                quantization = "gptq"
            elif precision == "awq":
                dtype = "auto"
                quantization = "awq"
            else:
                raise ValueError(f"Unknown precision: {precision}")

        llm = LLM(
            model=model_path,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )

        print(f"✓ Model loaded successfully")
        print_gpu_memory_status()
        return llm

    except Exception as e:
        print(f"✗ Failed to load {model_name} in {precision}: {e}")
        import traceback

        traceback.print_exc()
        return None


print("Model setup function loaded ✓")

# %% [markdown]
# ## Cell 8: Latency & Memory Benchmarking Function


# %%
def benchmark_latency_memory(
    llm: LLM,
    model_name: str,
    precision: str,
    num_runs: int = NUM_RUNS,
    num_warmup: int = NUM_WARMUP,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Benchmark vLLM model for latency and memory."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
    )

    latencies = []
    peak_memories = []
    avg_memories = []
    tokens_generated = []

    total_iterations = (num_runs + num_warmup + batch_size - 1) // batch_size

    print(f"\n{'=' * 60}")
    print(f"LATENCY & MEMORY BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Warmup: {num_warmup} | Benchmark: {num_runs} | Batch: {batch_size}")

    iteration_count = 0
    pbar = tqdm(total=num_runs + num_warmup, desc="Latency/Memory")

    for batch_idx in range(total_iterations):
        current_batch_size = min(batch_size, num_runs + num_warmup - iteration_count)
        prompts = [PROMPT] * current_batch_size

        clear_cache()
        torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        end_time = time.time()

        batch_latency = end_time - start_time
        peak_mem = get_peak_gpu_memory_mb()
        avg_mem = get_gpu_memory_mb()
        per_request_latency = batch_latency / current_batch_size
        batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        for i in range(current_batch_size):
            if iteration_count >= num_warmup:
                latencies.append(per_request_latency)
                peak_memories.append(peak_mem)
                avg_memories.append(avg_mem)
                tokens_generated.append(batch_tokens / current_batch_size)

            iteration_count += 1
            pbar.update(1)

            if iteration_count >= num_runs + num_warmup:
                break

        if iteration_count >= num_runs + num_warmup:
            break

    pbar.close()

    lat = np.array(latencies)
    pm = np.array(peak_memories)
    am = np.array(avg_memories)
    tg = np.array(tokens_generated)

    results = {
        "mean_latency_s": lat.mean(),
        "median_latency_s": np.median(lat),
        "p95_latency_s": np.percentile(lat, 95),
        "p99_latency_s": np.percentile(lat, 99),
        "std_latency_s": lat.std(),
        "mean_peak_mem_mb": pm.mean(),
        "mean_avg_mem_mb": am.mean(),
        "tokens_per_second": tg.mean() / lat.mean(),
    }

    print(
        f"✓ Latency: {results['mean_latency_s']:.4f}s | Memory: {results['mean_peak_mem_mb']:.2f}MB"
    )
    return results


print("Latency/Memory benchmark function loaded ✓")

# %% [markdown]
# ## Cell 9: Energy Tracking Function


# %%
def benchmark_energy(
    llm: LLM, model_name: str, precision: str, num_samples: int = ENERGY_SAMPLE_RUNS
) -> Dict:
    """Measure energy consumption using CodeCarbon."""

    print(f"\n{'=' * 60}")
    print(f"ENERGY CONSUMPTION BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Running {num_samples} inference samples with energy tracking...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
    )

    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the main causes of climate change?",
        "Solve: If x + 5 = 12, what is x?",
        "Describe the process of photosynthesis.",
    ]

    try:
        # Initialize energy tracker
        tracker = EmissionsTracker(
            project_name=f"slim_eval_{model_name.split('/')[-1]}_{precision}",
            output_dir=str(OUTPUT_DIR / "energy_logs"),
            log_level="warning",
            save_to_file=True,
        )

        # Start tracking
        tracker.start()
        start_time = time.time()

        # Run inference samples
        for i in tqdm(range(num_samples), desc="Energy tracking"):
            prompt = test_prompts[i % len(test_prompts)]
            prompts_batch = [prompt]
            outputs = llm.generate(prompts_batch, sampling_params)

        # Stop tracking
        end_time = time.time()
        emissions = tracker.stop()

        duration = end_time - start_time

        results = {
            "energy_kwh": emissions,  # kWh
            "energy_joules": emissions * 3600000,  # Convert kWh to Joules
            "duration_seconds": duration,
            "avg_power_watts": (emissions * 3600000 / duration) if duration > 0 else 0,
            "num_samples": num_samples,
            "energy_per_query_j": (emissions * 3600000 / num_samples)
            if num_samples > 0
            else 0,
        }

        print(
            f"✓ Energy: {results['energy_kwh'] * 1000:.4f} Wh | Avg Power: {results['avg_power_watts']:.2f}W"
        )
        return results

    except Exception as e:
        print(f"⚠️  Energy tracking failed: {e}")
        # Return fallback results with timing only
        return {
            "energy_kwh": 0,
            "energy_joules": 0,
            "duration_seconds": 0,
            "avg_power_watts": 0,
            "num_samples": 0,
            "energy_per_query_j": 0,
            "error": str(e),
        }


print("Energy benchmark function loaded ✓")

# %% [markdown]
# ## Cell 10: Accuracy Evaluation Function


# %%
def benchmark_accuracy(
    model_name: str,
    precision: str,
    tasks: List[str] = ACCURACY_TASKS,
    num_fewshot: int = NUM_FEW_SHOT,
    limit: Optional[int] = ACCURACY_LIMIT,
) -> Dict:
    """Evaluate model accuracy using lm-evaluation-harness."""

    print(f"\n{'=' * 60}")
    print(f"ACCURACY EVALUATION")
    print(f"{'=' * 60}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Few-shot: {num_fewshot}")
    if limit:
        print(f"Limit: {limit} examples per task")

    try:
        # Determine model path
        if precision == "fp16":
            model_path = model_name
        else:
            model_short_name = model_name.split("/")[-1]
            quantized_path = QUANTIZED_MODELS_DIR / f"{model_short_name}_{precision}"
            model_path = str(quantized_path) if quantized_path.exists() else model_name

        # Create VLLM model for lm-eval
        lm = VLLM_LM(
            pretrained=model_path,
            dtype="float16" if precision == "fp16" else "auto",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size="auto",
            limit=limit,
            log_samples=False,
        )

        # Extract metrics
        accuracy_results = {}
        for task_name, task_results in results["results"].items():
            # Find accuracy metric
            if "acc" in task_results:
                accuracy_results[f"{task_name}_accuracy"] = task_results["acc"]
            elif "exact_match" in task_results:
                accuracy_results[f"{task_name}_accuracy"] = task_results["exact_match"]
            elif "pass@1" in task_results:
                accuracy_results[f"{task_name}_accuracy"] = task_results["pass@1"]

        print(f"✓ Accuracy results:")
        for task, acc in accuracy_results.items():
            print(f"  {task}: {acc:.4f} ({acc * 100:.2f}%)")

        return accuracy_results

    except Exception as e:
        print(f"⚠️  Accuracy evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return {f"{task}_accuracy": 0 for task in tasks}


print("Accuracy benchmark function loaded ✓")

# %% [markdown]
# ## Cell 11: Main Benchmarking Loop


# %%
def run_complete_benchmark(model_name: str, precision: str) -> Dict:
    """Run complete benchmark: latency, memory, energy, and accuracy."""

    print(f"\n{'#' * 70}")
    print(f"COMPLETE BENCHMARK: {model_name} ({precision.upper()})")
    print(f"{'#' * 70}")

    results = {
        "model": model_name,
        "precision": precision,
        "timestamp": datetime.now().isoformat(),
    }

    # Setup model
    llm = setup_vllm_model(model_name, precision)
    if llm is None:
        print(f"✗ Model setup failed, skipping...")
        return None

    try:
        # 1. Latency & Memory
        latency_memory_results = benchmark_latency_memory(llm, model_name, precision)
        results.update(latency_memory_results)

        # 2. Energy (if enabled)
        if ENABLE_ENERGY_TRACKING:
            energy_results = benchmark_energy(llm, model_name, precision)
            results.update(energy_results)

        # Clean up vLLM model before accuracy eval
        del llm
        clear_cache()
        time.sleep(2)

        # 3. Accuracy
        accuracy_results = benchmark_accuracy(model_name, precision)
        results.update(accuracy_results)

        return results

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        clear_cache()


print("Complete benchmark function loaded ✓")

# %% [markdown]
# ## Cell 12: Initialize Results CSV

# %%
# Define all possible columns
base_columns = [
    "model",
    "precision",
    "timestamp",
    "mean_latency_s",
    "median_latency_s",
    "p95_latency_s",
    "p99_latency_s",
    "std_latency_s",
    "mean_peak_mem_mb",
    "mean_avg_mem_mb",
    "tokens_per_second",
]

energy_columns = [
    "energy_kwh",
    "energy_joules",
    "duration_seconds",
    "avg_power_watts",
    "energy_per_query_j",
]

accuracy_columns = [f"{task}_accuracy" for task in ACCURACY_TASKS]

all_columns = base_columns + energy_columns + accuracy_columns

if not RESULTS_CSV.exists():
    pd.DataFrame(columns=all_columns).to_csv(RESULTS_CSV, index=False)
    print(f"✓ Created results CSV: {RESULTS_CSV}")
else:
    print(f"✓ Results CSV exists: {RESULTS_CSV}")

# %% [markdown]
# ## Cell 13: Run All Benchmarks
# **This cell runs everything. Will take several hours!**

# %%
print(f"\n{'#' * 70}")
print("SLiM-Eval: COMPLETE BENCHMARK SUITE")
print(f"{'#' * 70}\n")
print(f"Models: {len(MODELS)}")
print(f"Precisions: {PRECISIONS}")
print(f"Total configs: {len(MODELS) * len(PRECISIONS)}")
print(f"Metrics: Latency, Memory, Energy, Accuracy")
print(f"Output: {RESULTS_CSV}")
print(f"\nEstimated time: ~{len(MODELS) * len(PRECISIONS) * 30} minutes")

all_results = []

for model_name in MODELS:
    for precision in PRECISIONS:
        config_id = f"{model_name.split('/')[-1]}_{precision}"

        # Run complete benchmark
        results = run_complete_benchmark(model_name, precision)

        if results:
            all_results.append(results)

            # Save incrementally
            pd.DataFrame([results]).to_csv(
                RESULTS_CSV, mode="a", header=False, index=False
            )
            print(f"\n✓ Results saved for {config_id}")

        # Cleanup between configs
        clear_cache()
        time.sleep(5)

print(f"\n{'#' * 70}")
print("ALL BENCHMARKS COMPLETE!")
print(f"{'#' * 70}")

# %% [markdown]
# ## Cell 14: Load and Display Results

# %%
# Load complete results
results_df = pd.read_csv(RESULTS_CSV)

print(f"\n{'#' * 70}")
print("COMPLETE RESULTS SUMMARY")
print(f"{'#' * 70}\n")

# Display key metrics
display_columns = [
    "model",
    "precision",
    "mean_latency_s",
    "mean_peak_mem_mb",
    "energy_kwh",
    "avg_power_watts",
] + [col for col in results_df.columns if "accuracy" in col]

print(results_df[display_columns].round(4).to_string(index=False))

print(f"\n✓ Full results: {RESULTS_CSV}")

# %% [markdown]
# ## Cell 15: Analysis - Efficiency vs Accuracy Trade-offs

# %%
# Calculate efficiency metrics
analysis_results = []

for model in results_df["model"].unique():
    model_data = results_df[results_df["model"] == model]
    fp16_data = model_data[model_data["precision"] == "fp16"]

    if len(fp16_data) == 0:
        continue

    fp16_latency = fp16_data["mean_latency_s"].values[0]
    fp16_memory = fp16_data["mean_peak_mem_mb"].values[0]
    fp16_energy = (
        fp16_data["energy_joules"].values[0]
        if "energy_joules" in fp16_data.columns
        else 0
    )

    for _, row in model_data.iterrows():
        if row["precision"] != "fp16":
            analysis_row = {
                "model": model.split("/")[-1],
                "precision": row["precision"],
                "speedup": fp16_latency / row["mean_latency_s"],
                "memory_reduction_pct": (1 - row["mean_peak_mem_mb"] / fp16_memory)
                * 100,
            }

            # Energy reduction
            if "energy_joules" in row and fp16_energy > 0:
                analysis_row["energy_reduction_pct"] = (
                    1 - row["energy_joules"] / fp16_energy
                ) * 100

            # Accuracy drop
            for task in ACCURACY_TASKS:
                acc_col = f"{task}_accuracy"
                if acc_col in fp16_data.columns and acc_col in row:
                    fp16_acc = fp16_data[acc_col].values[0]
                    quant_acc = row[acc_col]
                    if fp16_acc > 0:
                        analysis_row[f"{task}_acc_drop_pct"] = (
                            (fp16_acc - quant_acc) / fp16_acc
                        ) * 100

            analysis_results.append(analysis_row)

if analysis_results:
    analysis_df = pd.DataFrame(analysis_results)
    print("\n{'#'*70}")
    print("QUANTIZATION IMPACT ANALYSIS")
    print(f"{'#' * 70}\n")
    print(analysis_df.round(3).to_string(index=False))

    analysis_path = OUTPUT_DIR / "quantization_impact.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n✓ Analysis saved: {analysis_path}")

# %% [markdown]
# ## Cell 16: Visualization - Pareto Frontiers

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Latency vs Memory
ax1 = axes[0, 0]
for precision in results_df["precision"].unique():
    data = results_df[results_df["precision"] == precision]
    ax1.scatter(
        data["mean_latency_s"],
        data["mean_peak_mem_mb"],
        label=precision.upper(),
        s=150,
        alpha=0.7,
    )
ax1.set_xlabel("Latency (seconds)")
ax1.set_ylabel("Peak Memory (MB)")
ax1.set_title("Latency vs Memory")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Energy vs Accuracy (if available)
if "energy_kwh" in results_df.columns and any(
    "accuracy" in col for col in results_df.columns
):
    ax2 = axes[0, 1]
    acc_col = [col for col in results_df.columns if "accuracy" in col][0]
    for precision in results_df["precision"].unique():
        data = results_df[results_df["precision"] == precision]
        ax2.scatter(
            data["energy_kwh"], data[acc_col], label=precision.upper(), s=150, alpha=0.7
        )
    ax2.set_xlabel("Energy (kWh)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Energy vs Accuracy Trade-off")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Throughput comparison
ax3 = axes[1, 0]
models_short = results_df["model"].str.split("/").str[-1]
for precision in results_df["precision"].unique():
    data = results_df[results_df["precision"] == precision]
    indices = data.index
    ax3.bar(
        [models_short[i] for i in indices],
        data["tokens_per_second"],
        label=precision.upper(),
        alpha=0.7,
    )
ax3.set_xlabel("Model")
ax3.set_ylabel("Tokens/Second")
ax3.set_title("Throughput by Model and Precision")
ax3.legend()
ax3.tick_params(axis="x", rotation=45)

# Plot 4: Accuracy comparison across tasks
ax4 = axes[1, 1]
accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
if accuracy_cols:
    # Average accuracy across all tasks
    results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)
    for precision in results_df["precision"].unique():
        data = results_df[results_df["precision"] == precision]
        indices = data.index
        ax4.bar(
            [models_short[i] for i in indices],
            data["avg_accuracy"],
            label=precision.upper(),
            alpha=0.7,
        )
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Average Accuracy")
    ax4.set_title("Average Accuracy Across Tasks")
    ax4.legend()
    ax4.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pareto_frontiers.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"✓ Visualizations saved: {OUTPUT_DIR / 'pareto_frontiers.png'}")

# %% [markdown]
# ## Cell 17: Summary Statistics Tables

# %%
# Create comprehensive summary
summary_stats = (
    results_df.groupby(["precision"])
    .agg(
        {
            "mean_latency_s": ["mean", "std", "min", "max"],
            "mean_peak_mem_mb": ["mean", "std", "min", "max"],
            "tokens_per_second": ["mean", "std", "min", "max"],
        }
    )
    .round(4)
)

if "energy_kwh" in results_df.columns:
    energy_stats = (
        results_df.groupby(["precision"])
        .agg(
            {
                "energy_kwh": ["mean", "std"],
                "avg_power_watts": ["mean", "std"],
            }
        )
        .round(4)
    )
    summary_stats = pd.concat([summary_stats, energy_stats], axis=1)

print("\n{'#'*70}")
print("SUMMARY STATISTICS BY PRECISION")
print(f"{'#' * 70}\n")
print(summary_stats)

summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv")
print(f"\n✓ Summary saved: {OUTPUT_DIR / 'summary_statistics.csv'}")

# %% [markdown]
# ## Cell 18: Paper-Ready Results Table

# %%
# Create formatted table for academic papers
paper_columns = [
    "model",
    "precision",
    "mean_latency_s",
    "mean_peak_mem_mb",
    "tokens_per_second",
]

if "energy_kwh" in results_df.columns:
    paper_columns.append("energy_kwh")

accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
if accuracy_cols:
    paper_columns.extend(accuracy_cols[:3])  # Include first 3 accuracy metrics

paper_table = results_df[paper_columns].copy()
paper_table["model"] = paper_table["model"].str.split("/").str[-1]

# Rename columns for readability
rename_map = {
    "mean_latency_s": "Latency (s)",
    "mean_peak_mem_mb": "Memory (MB)",
    "tokens_per_second": "Tokens/s",
    "energy_kwh": "Energy (kWh)",
}
paper_table = paper_table.rename(columns=rename_map)
paper_table = paper_table.round(4)

print("\n{'#'*70}")
print("PAPER-READY RESULTS TABLE")
print(f"{'#' * 70}\n")
print(paper_table.to_string(index=False))

# Save as CSV and LaTeX
paper_table.to_csv(OUTPUT_DIR / "paper_results.csv", index=False)
latex_table = paper_table.to_latex(index=False, float_format="%.4f")
with open(OUTPUT_DIR / "paper_results.tex", "w") as f:
    f.write(latex_table)

print(f"\n✓ Paper table saved: {OUTPUT_DIR / 'paper_results.csv'}")
print(f"✓ LaTeX table saved: {OUTPUT_DIR / 'paper_results.tex'}")

# %% [markdown]
# ## Cell 19: Export JSON Results (for sharing/archiving)

# %%
# Export all results as JSON for easy sharing
json_output = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "models": MODELS,
        "precisions": PRECISIONS,
        "accuracy_tasks": ACCURACY_TASKS,
        "num_runs": NUM_RUNS,
        "batch_size": BATCH_SIZE,
    },
    "results": results_df.to_dict(orient="records"),
}

if analysis_results:
    json_output["analysis"] = analysis_df.to_dict(orient="records")

json_path = OUTPUT_DIR / "complete_results.json"
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)

print(f"✓ JSON results saved: {json_path}")

# %% [markdown]
# ## Cell 20: Generate Executive Summary Report

# %%
# Create a text report summarizing key findings
report_lines = [
    "=" * 70,
    "SLiM-EVAL: EXECUTIVE SUMMARY REPORT",
    "=" * 70,
    f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"\nModels Evaluated: {len(results_df['model'].unique())}",
    f"Precision Modes: {', '.join(results_df['precision'].unique())}",
    f"Total Configurations: {len(results_df)}",
    "\n" + "=" * 70,
    "\nKEY FINDINGS:",
    "=" * 70,
]

# Find best performers
if len(results_df) > 0:
    # Fastest model
    fastest = results_df.loc[results_df["mean_latency_s"].idxmin()]
    report_lines.append(f"\n1. FASTEST MODEL:")
    report_lines.append(
        f"   {fastest['model'].split('/')[-1]} ({fastest['precision']})"
    )
    report_lines.append(f"   Latency: {fastest['mean_latency_s']:.4f}s")

    # Most memory efficient
    mem_efficient = results_df.loc[results_df["mean_peak_mem_mb"].idxmin()]
    report_lines.append(f"\n2. MOST MEMORY EFFICIENT:")
    report_lines.append(
        f"   {mem_efficient['model'].split('/')[-1]} ({mem_efficient['precision']})"
    )
    report_lines.append(f"   Memory: {mem_efficient['mean_peak_mem_mb']:.2f} MB")

    # Highest throughput
    highest_throughput = results_df.loc[results_df["tokens_per_second"].idxmax()]
    report_lines.append(f"\n3. HIGHEST THROUGHPUT:")
    report_lines.append(
        f"   {highest_throughput['model'].split('/')[-1]} ({highest_throughput['precision']})"
    )
    report_lines.append(
        f"   Throughput: {highest_throughput['tokens_per_second']:.2f} tokens/s"
    )

    # Best accuracy (if available)
    accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
    if accuracy_cols:
        results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)
        best_accuracy = results_df.loc[results_df["avg_accuracy"].idxmax()]
        report_lines.append(f"\n4. BEST AVERAGE ACCURACY:")
        report_lines.append(
            f"   {best_accuracy['model'].split('/')[-1]} ({best_accuracy['precision']})"
        )
        report_lines.append(f"   Avg Accuracy: {best_accuracy['avg_accuracy']:.4f}")

    # Quantization impact
    if analysis_results:
        report_lines.append(f"\n" + "=" * 70)
        report_lines.append("QUANTIZATION IMPACT:")
        report_lines.append("=" * 70)

        avg_speedup = analysis_df["speedup"].mean()
        avg_mem_reduction = analysis_df["memory_reduction_pct"].mean()

        report_lines.append(f"\nINT8/INT4 Quantization Effects (Average):")
        report_lines.append(f"  • Speedup: {avg_speedup:.2f}x")
        report_lines.append(f"  • Memory Reduction: {avg_mem_reduction:.1f}%")

        if "energy_reduction_pct" in analysis_df.columns:
            avg_energy_reduction = analysis_df["energy_reduction_pct"].mean()
            report_lines.append(f"  • Energy Reduction: {avg_energy_reduction:.1f}%")

        acc_drop_cols = [col for col in analysis_df.columns if "acc_drop" in col]
        if acc_drop_cols:
            avg_acc_drop = analysis_df[acc_drop_cols].mean().mean()
            report_lines.append(f"  • Average Accuracy Drop: {avg_acc_drop:.2f}%")

report_lines.append("\n" + "=" * 70)
report_lines.append("END OF REPORT")
report_lines.append("=" * 70)

report_text = "\n".join(report_lines)
print(report_text)

# Save report
report_path = OUTPUT_DIR / "executive_summary.txt"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\n✓ Executive summary saved: {report_path}")

# %% [markdown]
# ## Cell 21: Quick Single Model Test (For Debugging)

# %%
# Uncomment to test a single model quickly
# TEST_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# TEST_PRECISION = "fp16"
#
# print(f"Quick test: {TEST_MODEL} in {TEST_PRECISION}")
# results = run_complete_benchmark(TEST_MODEL, TEST_PRECISION)
# if results:
#     print("\nTest Results:")
#     for key, value in results.items():
#         print(f"  {key}: {value}")

print("Quick test cell ready (not executed)")

# %% [markdown]
# ---
# ## Summary of Complete Framework
#
# ### Metrics Tracked:
# 1. **Latency** - Mean, median, P95, P99 inference time
# 2. **Memory** - Peak and average GPU memory usage
# 3. **Energy** - Power consumption and energy per query
# 4. **Accuracy** - Task performance on MMLU, GSM8K, HellaSwag, etc.
#
# ### Workflow:
# 1. **Setup** - Configure models, precisions, benchmarks
# 2. **Quantize** (optional) - Pre-quantize models with llm-compressor
# 3. **Benchmark** - Run latency, memory, energy, and accuracy tests
# 4. **Analyze** - Calculate trade-offs and efficiency metrics
# 5. **Visualize** - Generate plots and Pareto frontiers
# 6. **Report** - Export results in CSV, JSON, LaTeX formats
#
# ### Next Steps:
# - Add carbon emissions tracking (multiply energy by grid carbon intensity)
# - Integrate additional benchmarks (BBH, MATH, MedQA, LegalBench)
# - Add throughput benchmarking under different batch sizes
# - Implement automated Pareto frontier analysis
# - Add statistical significance testing for accuracy differences
