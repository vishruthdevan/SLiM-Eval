"""
SLiM-Eval: Small Language Model Evaluation Framework
Latency and Memory Benchmarking with vLLM and llm-compressor
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# =============================================================================
# CONFIGURATION
# =============================================================================

# Editable model list - add/remove models as needed
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",  # Phi-3 3.8B
    "microsoft/Phi-3.5-mini-instruct",  # Phi-3.5 3.8B
    "google/gemma-2-2b-it",  # Gemma 2 2B
    "meta-llama/Llama-3.2-3B",  # Llama 3.2 3B
    "Qwen/Qwen2.5-3B-Instruct",  # Qwen2.5 3B
    "mistralai/Mistral-7B-Instruct-v0.3",  # Mistral 7B
]

# Benchmark configuration
NUM_RUNS = 1000
NUM_WARMUP = 10  # Warmup runs to exclude from metrics
MAX_NEW_TOKENS = 32
BATCH_SIZE = 32  # vLLM batching for efficiency
PROMPT = "Explain one interesting fact about large language models."

# Quantization precisions to test
PRECISIONS = ["fp16", "int8", "int4", "gptq", "awq"]

# Output configuration
OUTPUT_DIR = Path("slim_eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUTPUT_DIR / "latency_memory_results.csv"

# =============================================================================
# DEVICE SETUP
# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for vLLM. Please run on GPU.")

DEVICE = "cuda"
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# Verify PyTorch version
pytorch_version = tuple(map(int, torch.__version__.split(".")[:2]))
if pytorch_version < (2, 0):
    print(f"WARNING: PyTorch {torch.__version__} detected. Recommended: 2.3+")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


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


def setup_vllm_model(model_name: str, precision: str) -> Optional[LLM]:
    """
    Setup vLLM model with specified precision.

    Args:
        model_name: HuggingFace model identifier
        precision: One of ["fp16", "int8", "int4", "gptq", "awq"]

    Returns:
        Loaded vLLM model or None if setup fails
    """
    clear_cache()

    try:
        print(f"\n{'=' * 60}")
        print(f"Loading {model_name} in {precision.upper()} precision...")
        print(f"{'=' * 60}")

        # Configure dtype based on precision
        if precision == "fp16":
            dtype = "float16"
            quantization = None
        elif precision == "int8":
            dtype = "auto"
            quantization = "int8"  # vLLM native int8
        elif precision == "int4":
            dtype = "auto"
            quantization = "int4"  # Requires llm-compressor quantized model
        elif precision == "gptq":
            dtype = "auto"
            quantization = "gptq"
        elif precision == "awq":
            dtype = "auto"
            quantization = "awq"
        else:
            raise ValueError(f"Unknown precision: {precision}")

        # Initialize vLLM model
        # Note: For int4/gptq/awq, you may need to use pre-quantized models
        # or quantize models first using llm-compressor
        llm = LLM(
            model=model_name,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=0.9,
            max_model_len=2048,  # Adjust based on your needs
            tensor_parallel_size=1,  # Increase if multiple GPUs
            trust_remote_code=True,
        )

        print(f"✓ Model loaded successfully")
        return llm

    except Exception as e:
        print(f"✗ Failed to load {model_name} in {precision}: {e}")
        return None


def benchmark_vllm_model(
    llm: LLM,
    model_name: str,
    precision: str,
    num_runs: int = NUM_RUNS,
    num_warmup: int = NUM_WARMUP,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """
    Benchmark vLLM model for latency and memory.

    Args:
        llm: Loaded vLLM model
        model_name: Model identifier
        precision: Precision mode
        num_runs: Number of benchmark runs
        num_warmup: Number of warmup runs to exclude
        batch_size: Batch size for inference

    Returns:
        Dictionary of benchmark results
    """
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

    print(f"\nRunning {num_warmup} warmup + {num_runs} benchmark iterations...")
    print(f"Batch size: {batch_size}")

    iteration_count = 0
    pbar = tqdm(total=num_runs + num_warmup, desc="Benchmarking")

    for batch_idx in range(total_iterations):
        # Prepare batch of prompts
        current_batch_size = min(batch_size, num_runs + num_warmup - iteration_count)
        prompts = [PROMPT] * current_batch_size

        # Clear cache before each batch
        clear_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure latency
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        end_time = time.time()

        batch_latency = end_time - start_time

        # Memory metrics
        peak_mem = get_peak_gpu_memory_mb()
        avg_mem = get_gpu_memory_mb()

        # Per-request metrics (amortize batch time)
        per_request_latency = batch_latency / current_batch_size

        # Count tokens generated
        batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        # Store metrics (skip warmup runs)
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

    # Calculate statistics
    lat = np.array(latencies)
    pm = np.array(peak_memories)
    am = np.array(avg_memories)
    tg = np.array(tokens_generated)

    results = {
        "model": model_name,
        "precision": precision,
        "num_runs": len(latencies),
        "mean_latency_s": lat.mean(),
        "median_latency_s": np.median(lat),
        "p95_latency_s": np.percentile(lat, 95),
        "p99_latency_s": np.percentile(lat, 99),
        "std_latency_s": lat.std(),
        "mean_peak_mem_mb": pm.mean(),
        "mean_avg_mem_mb": am.mean(),
        "mean_tokens_generated": tg.mean(),
        "tokens_per_second": tg.mean() / lat.mean(),
    }

    return results


def print_results(results: Dict):
    """Pretty print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"Results: {results['model']} ({results['precision'].upper()})")
    print(f"{'=' * 60}")
    print(f"Runs:                {results['num_runs']}")
    print(f"Mean Latency:        {results['mean_latency_s']:.4f} s")
    print(f"Median Latency:      {results['median_latency_s']:.4f} s")
    print(f"P95 Latency:         {results['p95_latency_s']:.4f} s")
    print(f"P99 Latency:         {results['p99_latency_s']:.4f} s")
    print(f"Std Latency:         {results['std_latency_s']:.4f} s")
    print(f"Peak Memory:         {results['mean_peak_mem_mb']:.2f} MB")
    print(f"Avg Memory:          {results['mean_avg_mem_mb']:.2f} MB")
    print(f"Tokens/Second:       {results['tokens_per_second']:.2f}")
    print(f"{'=' * 60}\n")


# =============================================================================
# MAIN BENCHMARK LOOP
# =============================================================================


def main():
    """Main benchmarking function."""
    print(f"\n{'#' * 60}")
    print("SLiM-Eval: Latency & Memory Benchmark")
    print(f"{'#' * 60}\n")
    print(f"Models to evaluate: {len(MODELS)}")
    print(f"Precisions to test: {PRECISIONS}")
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Initialize results CSV
    if not CSV_PATH.exists():
        pd.DataFrame(
            columns=[
                "model",
                "precision",
                "num_runs",
                "mean_latency_s",
                "median_latency_s",
                "p95_latency_s",
                "p99_latency_s",
                "std_latency_s",
                "mean_peak_mem_mb",
                "mean_avg_mem_mb",
                "mean_tokens_generated",
                "tokens_per_second",
            ]
        ).to_csv(CSV_PATH, index=False)

    all_results = []

    # Benchmark each model-precision combination
    for model_name in MODELS:
        for precision in PRECISIONS:
            config_id = f"{model_name.split('/')[-1]}_{precision}"
            print(f"\n{'#' * 60}")
            print(f"Config: {config_id}")
            print(f"{'#' * 60}")

            # Setup model
            llm = setup_vllm_model(model_name, precision)
            if llm is None:
                print(f"Skipping {config_id} due to setup failure")
                continue

            try:
                # Run benchmark
                results = benchmark_vllm_model(
                    llm=llm,
                    model_name=model_name,
                    precision=precision,
                    num_runs=NUM_RUNS,
                    num_warmup=NUM_WARMUP,
                    batch_size=BATCH_SIZE,
                )

                # Store and print results
                all_results.append(results)
                print_results(results)

                # Save to CSV incrementally
                pd.DataFrame([results]).to_csv(
                    CSV_PATH, mode="a", header=False, index=False
                )
                print(f"✓ Results appended to {CSV_PATH}")

            except Exception as e:
                print(f"✗ Benchmark failed for {config_id}: {e}")

            finally:
                # Cleanup
                del llm
                clear_cache()
                time.sleep(2)  # Brief pause between configs

    # Final summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'#' * 60}")
        print("FINAL SUMMARY")
        print(f"{'#' * 60}\n")
        print(df.to_string(index=False))
        print(f"\n✓ Complete results saved to {CSV_PATH}")

        # Save detailed summary
        summary_path = OUTPUT_DIR / "benchmark_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"✓ Summary saved to {summary_path}")
    else:
        print("\n✗ No successful benchmarks completed")


if __name__ == "__main__":
    main()
