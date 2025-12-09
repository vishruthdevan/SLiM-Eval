#!/usr/bin/env python3
"""
LLM Evaluation Framework for M3 Max
Evaluates models on MMLU-Pro, GSM8K, and HumanEval
Measures accuracy and energy consumption
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import traceback

# ============================================================================
# CONFIGURATION
# ============================================================================

# Primary model to evaluate
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Alternative models (commented out - uncomment to use)
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Phi-3-mini 3.8B
# MODEL_NAME = "google/gemma-2-2b"  # Gemma 2-2B
# MODEL_NAME = "Qwen/Qwen2.5-3B"  # Qwen2.5-3B
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Mistral 7B

# Benchmarks to evaluate
TASKS = [
    "mmlu_pro",      # MMLU-Pro (multiple choice reasoning)
    "gsm8k",         # GSM8K (math word problems)
    "humaneval",     # HumanEval (code generation)
]

# Evaluation parameters
NUM_FEW_SHOT = 5  # Number of few-shot examples
BATCH_SIZE = 1    # Keep at 1 for energy measurements
LIMIT = None      # Set to small number (e.g., 10) for quick testing

# Energy monitoring
ENABLE_ENERGY_MONITORING = False  # Set to True once we fix powermetrics

# Output directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: str, device: str = "mps"):
    """
    Load model and tokenizer with MPS (Metal Performance Shaders) support.
    
    Args:
        model_name: Hugging Face model identifier
        device: Device to load model on ("mps" for M3 Max, "cpu" as fallback)
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n{'='*70}")
    print(f"Loading Model: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Check device availability
        if device == "mps" and not torch.backends.mps.is_available():
            print("⚠ MPS not available, falling back to CPU")
            device = "cpu"
        
        print(f"Target device: {device}")
        
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   ✓ Tokenizer loaded")
        
        # Load model
        print("\n2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"   ✓ Model loaded to {device}")
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n3. Model Information:")
        print(f"   • Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        print(f"   • Dtype: {model.dtype}")
        print(f"   • Device: {next(model.parameters()).device}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def run_benchmark(model_name: str, tasks: list, num_fewshot: int = 5, 
                  batch_size: int = 1, limit: int = None):
    """
    Run evaluation on specified tasks using lm-evaluation-harness.
    
    Args:
        model_name: Hugging Face model identifier
        tasks: List of task names to evaluate
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        limit: Limit number of examples (None for all)
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*70}")
    print(f"Running Evaluation")
    print(f"{'='*70}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Few-shot: {num_fewshot}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limit: {limit} examples per task")
    
    try:
        # Create LM object
        lm = HFLM(
            pretrained=model_name,
            dtype="float16",
            device="mps" if torch.backends.mps.is_available() else "cpu",
            batch_size=batch_size,
        )
        
        # Run evaluation
        print("\nStarting evaluation...")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
            log_samples=False,
            confirm_run_unsafe_code=True,  # Enable HumanEval
        )
        
        return results
        
    except Exception as e:
        print(f"\n✗ Evaluation error: {e}")
        traceback.print_exc()
        return None

def measure_energy_simple(model, tokenizer, num_samples: int = 100):
    """
    Simple timing-based measurement (placeholder for energy monitoring).
    
    Args:
        model: Loaded language model
        tokenizer: Model tokenizer
        num_samples: Number of inference runs
    
    Returns:
        dict: Timing statistics
    """
    print(f"\n{'='*70}")
    print(f"Measuring Inference Performance")
    print(f"{'='*70}")
    print(f"Running {num_samples} inference samples...")
    
    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the main causes of climate change?",
        "Solve: If x + 5 = 12, what is x?",
        "Describe the process of photosynthesis.",
    ]
    
    start_time = time.time()
    latencies = []
    
    try:
        # Run inference samples
        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]
            
            sample_start = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            
            sample_end = time.time()
            latencies.append(sample_end - sample_start)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_samples} samples...")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"  ✓ All {num_samples} samples completed")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        
        results = {
            'duration_seconds': duration,
            'num_samples': num_samples,
            'avg_latency_seconds': avg_latency,
            'throughput_queries_per_second': num_samples / duration,
            'note': 'Energy monitoring to be added - currently showing timing only'
        }
        
        print(f"\n📊 Performance Results:")
        print(f"   • Total duration: {duration:.2f} s")
        print(f"   • Avg latency: {avg_latency:.4f} s")
        print(f"   • Throughput: {results['throughput_queries_per_second']:.2f} queries/s")
        
        return results
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# RESULTS PROCESSING
# ============================================================================

def extract_metrics(results: dict):
    """
    Extract key metrics from lm-eval results.
    
    Args:
        results: Raw results from lm-evaluation-harness
    
    Returns:
        dict: Processed metrics
    """
    metrics = {}
    
    if not results or 'results' not in results:
        return metrics
    
    for task_name, task_results in results['results'].items():
        task_metrics = {}
        
        # Extract accuracy metrics
        if 'acc' in task_results:
            task_metrics['accuracy'] = task_results['acc']
        elif 'exact_match' in task_results:
            task_metrics['accuracy'] = task_results['exact_match']
        elif 'pass@1' in task_results:
            task_metrics['accuracy'] = task_results['pass@1']
        
        # Extract other available metrics
        for key, value in task_results.items():
            if key not in ['alias', 'samples'] and isinstance(value, (int, float)):
                task_metrics[key] = value
        
        metrics[task_name] = task_metrics
    
    return metrics

def save_results(model_name: str, eval_results: dict, energy_results: dict):
    """
    Save evaluation results to JSON file.
    
    Args:
        model_name: Model identifier
        eval_results: Evaluation metrics
        energy_results: Energy/performance metrics
    """
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.replace("/", "_")
    filename = RESULTS_DIR / f"{model_slug}_{timestamp}.json"
    
    # Compile full results
    output = {
        'model': model_name,
        'timestamp': timestamp,
        'evaluation': extract_metrics(eval_results) if eval_results else {},
        'performance': energy_results if energy_results else {},
        'config': {
            'tasks': TASKS,
            'num_fewshot': NUM_FEW_SHOT,
            'batch_size': BATCH_SIZE,
            'limit': LIMIT,
        }
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    return filename

def print_summary(eval_results: dict, energy_results: dict):
    """
    Print formatted summary of results.
    
    Args:
        eval_results: Evaluation metrics
        energy_results: Energy/performance metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    if eval_results and 'results' in eval_results:
        print("\n📊 Task Accuracy:")
        for task_name, task_results in eval_results['results'].items():
            # Find accuracy metric
            acc = None
            if 'acc' in task_results:
                acc = task_results['acc']
            elif 'exact_match' in task_results:
                acc = task_results['exact_match']
            elif 'pass@1' in task_results:
                acc = task_results['pass@1']
            
            if acc is not None:
                print(f"   • {task_name:20s}: {acc:.4f} ({acc*100:.2f}%)")
            else:
                print(f"   • {task_name:20s}: No accuracy metric found")
    
    if energy_results:
        print(f"\n⚡ Performance Metrics:")
        if 'avg_latency_seconds' in energy_results:
            print(f"   • Avg latency: {energy_results['avg_latency_seconds']:.4f} s")
            print(f"   • Throughput: {energy_results['throughput_queries_per_second']:.2f} queries/s")
        if 'note' in energy_results:
            print(f"   ℹ️  {energy_results['note']}")
    
    print(f"\n{'='*70}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print(f"\n{'#'*70}")
    print(f"# LLM EVALUATION FRAMEWORK - M3 MAX")
    print(f"{'#'*70}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, tokenizer = load_model(MODEL_NAME)
    
    # Run benchmark evaluation
    print(f"\n{'='*70}")
    print("PHASE 1: Task Accuracy Evaluation")
    print(f"{'='*70}")
    eval_results = run_benchmark(
        model_name=MODEL_NAME,
        tasks=TASKS,
        num_fewshot=NUM_FEW_SHOT,
        batch_size=BATCH_SIZE,
        limit=LIMIT
    )
    
    # Measure performance (timing for now)
    print(f"\n{'='*70}")
    print("PHASE 2: Performance Measurement")
    print(f"{'='*70}")
    
    if ENABLE_ENERGY_MONITORING:
        print("⚠️  Energy monitoring not yet implemented")
        energy_results = None
    else:
        energy_results = measure_energy_simple(
            model=model,
            tokenizer=tokenizer,
            num_samples=100
        )
    
    # Save and display results
    save_results(MODEL_NAME, eval_results, energy_results)
    print_summary(eval_results, energy_results)
    
    print("✓ Evaluation complete!")

if __name__ == "__main__":
    main()