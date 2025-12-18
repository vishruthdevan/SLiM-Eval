#!/usr/bin/env python3
"""
SLiM-Eval: Complete Small Language Model Evaluation Framework
Tracks: Latency, Memory, Energy, and Accuracy

This script provides a production-ready CLI tool for benchmarking small language models
with a specified precision and metrics.
"""

import argparse
import gc
import json
import logging
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.distributed as dist

# Import pynvml for GPU power tracking
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Import quantization libraries
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation

# Import lm-eval for accuracy
from lm_eval import evaluator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SLiMEvaluator:
    """Main evaluator class for SLiM-Eval benchmarking."""

    def __init__(self, args):
        """Initialize evaluator with configuration."""
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.quantized_models_dir = Path(args.quantized_models_dir)
        self.quantized_models_dir.mkdir(exist_ok=True)

        self.results_csv = self.output_dir / "complete_results.csv"

        # Setup file logging
        log_file = (
            self.output_dir
            / f"slim_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info("SLiM-Eval initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models: {args.models}")
        logger.info(f"Precision: {args.precision}")

        # Log GPU info
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        else:
            logger.warning("CUDA not available")

        # Quantization configurations
        self.quantization_configs = {
            "int8": {
                "recipe": [
                    SmoothQuantModifier(smoothing_strength=0.5),
                    GPTQModifier(
                        targets="Linear",
                        scheme="W8A8",
                        ignore=["lm_head", "embed_tokens", "norm", "rotary_emb"],
                    ),
                ],
                "method": "gptq_smoothquant",
                "scheme_name": "SmoothQuant+GPTQ (W8A8)",
            },
            "int4": {
                "recipe": [
                    SmoothQuantModifier(smoothing_strength=0.5),
                    GPTQModifier(
                        targets="Linear",
                        scheme="W4A16",
                        ignore=["lm_head", "embed_tokens", "norm", "rotary_emb"],
                    ),
                ],
                "method": "gptq_smoothquant",
                "scheme_name": "SmoothQuant+GPTQ (W4A16)",
            },
            "gptq": {
                "recipe": GPTQModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head", "embed_tokens", "norm", "rotary_emb"],
                ),
                "method": "gptq_only",
                "scheme_name": "GPTQ (W4A16)",
            },
        }

    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection."""
        # Destroy distributed process group if initialized
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.debug(f"Error destroying process group: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_gpu_memory_mb() -> float:
        """Get current GPU memory usage in MB using nvidia-smi."""
        if torch.cuda.is_available():
            try:
                if PYNVML_AVAILABLE:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    return info.used / (1024**2)
                else:
                    # Fallback to torch
                    return torch.cuda.memory_allocated() / (1024**2)
            except Exception:
                return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    @staticmethod
    def get_peak_gpu_memory_mb() -> float:
        """Get peak GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0

    @staticmethod
    def get_model_size(model_path: str) -> Dict[str, float]:
        """Get model size in parameters and GB."""
        try:
            # Load config to get parameter count
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Try to get number of parameters from config
            # Different models store this differently
            num_params = 0
            if hasattr(config, "num_parameters"):
                num_params = config.num_parameters
            elif hasattr(config, "n_params"):
                num_params = config.n_params
            else:
                # Estimate from hidden size and layers
                if hasattr(config, "hidden_size") and hasattr(
                    config, "num_hidden_layers"
                ):
                    hidden_size = config.hidden_size
                    num_layers = config.num_hidden_layers
                    vocab_size = getattr(config, "vocab_size", 32000)

                    # Rough estimate: embeddings + layers + LM head
                    embed_params = vocab_size * hidden_size
                    layer_params = num_layers * (
                        12 * hidden_size * hidden_size
                    )  # Approximate
                    head_params = vocab_size * hidden_size
                    num_params = embed_params + layer_params + head_params

            # Convert to billions
            num_params_b = num_params / 1e9

            # Estimate size in GB (FP16: 2 bytes per parameter)
            size_gb_fp16 = (num_params * 2) / (1024**3)

            return {
                "num_parameters": num_params,
                "num_parameters_b": num_params_b,
                "size_gb_fp16": size_gb_fp16,
            }
        except Exception as e:
            logger.warning(f"Failed to get model size for {model_path}: {e}")
            return {
                "num_parameters": 0,
                "num_parameters_b": 0,
                "size_gb_fp16": 0,
            }

    def quantize_model(self, model_name: str, precision: str, output_dir: Path):
        """Quantize a model using llm-compressor with SmoothQuant + GPTQ."""
        logger.info("=" * 60)
        logger.info(f"Quantizing {model_name} to {precision.upper()}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        if output_dir.exists() and (output_dir / "config.json").exists():
            logger.info("Already quantized, skipping...")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if precision not in self.quantization_configs:
                logger.error(f"Unsupported precision: {precision}")
                return

            # Load model and tokenizer
            logger.info("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Set pad token if not set (required for proper attention masks)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

            # Load and preprocess calibration dataset
            logger.info(f"Loading calibration dataset: {self.args.calibration_dataset}")
            ds = load_dataset(
                self.args.calibration_dataset,
                split=f"{self.args.calibration_split}[:{self.args.num_calibration_samples}]",
            )
            ds = ds.shuffle(seed=42)

            # Preprocess with chat template (with fallback for models without chat template)
            def preprocess(example):
                try:
                    return {
                        "text": tokenizer.apply_chat_template(
                            example["messages"],
                            tokenize=False,
                        )
                    }
                except Exception:
                    messages = example["messages"]
                    text_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        text_parts.append(f"{role}: {content}")
                    return {"text": "\n".join(text_parts)}

            ds = ds.map(preprocess)

            # Tokenize
            def tokenize(sample):
                return tokenizer(
                    sample["text"],
                    padding=False,
                    max_length=self.args.max_sequence_length,
                    truncation=True,
                    add_special_tokens=False,
                )

            ds = ds.map(tokenize, remove_columns=ds.column_names)

            # Get quantization recipe
            quant_config = self.quantization_configs[precision]
            recipe = quant_config["recipe"]

            logger.info(f"Applying quantization recipe: {quant_config['method']}")

            # Apply quantization
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=self.args.max_sequence_length,
                num_calibration_samples=self.args.num_calibration_samples,
            )

            # Verify with sample generation
            logger.info("Verifying quantized model with sample generation...")
            dispatch_for_generation(model)
            inputs = tokenizer(
                "Hello my name is",
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(model.device)
            output = model.generate(
                **inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id
            )
            logger.info(
                f"Sample output: {tokenizer.decode(output[0], skip_special_tokens=True)}"
            )

            # Save quantized model
            logger.info(f"Saving to {output_dir}...")
            model.save_pretrained(output_dir, save_compressed=True)
            tokenizer.save_pretrained(output_dir)

            logger.info(f"Quantization complete: {output_dir}")

            # Cleanup
            del model
            self.clear_cache()

        except Exception as e:
            logger.error(f"Quantization failed: {e}", exc_info=True)

    def setup_vllm_model(
        self, model_name: str, precision: str, use_quantized_dir: bool = True
    ) -> Optional[LLM]:
        """Setup vLLM model with specified precision."""
        self.clear_cache()

        try:
            logger.info("=" * 60)
            logger.info(f"Loading {model_name} in {precision.upper()} precision...")
            logger.info("=" * 60)

            if precision == "fp16":
                model_path = model_name
                model_lower = model_name.lower()

                # Model-specific dtype based on HuggingFace documentation
                # Use bfloat16 for: Llama, Mistral, Gemma, Phi, Qwen
                if any(
                    x in model_lower
                    for x in ["gemma", "llama", "mistral", "phi", "qwen"]
                ):
                    dtype = "bfloat16"
                    logger.info(
                        f"Using bfloat16 for {model_name} (recommended by model authors)"
                    )
                else:
                    dtype = "float16"
                quantization = None
            else:
                model_short_name = model_name.split("/")[-1]
                quantized_path = (
                    self.quantized_models_dir / f"{model_short_name}_{precision}"
                )

                if use_quantized_dir and quantized_path.exists():
                    model_path = str(quantized_path)
                    dtype = "auto"
                    quantization = None
                    logger.info(f"Using pre-quantized model from: {quantized_path}")
                    logger.info(
                        f"  Method: {self.quantization_configs[precision]['method']}"
                    )
                elif use_quantized_dir:
                    logger.info(f"Pre-quantized model not found at: {quantized_path}")
                    logger.info("Running quantization with llm-compressor...")
                    self.quantize_model(model_name, precision, quantized_path)

                    if quantized_path.exists():
                        model_path = str(quantized_path)
                        dtype = "auto"
                        quantization = None
                        logger.info(
                            f"Using newly quantized model from: {quantized_path}"
                        )
                    else:
                        logger.warning(
                            "Quantization failed, falling back to on-the-fly quantization"
                        )
                        model_path = model_name
                        dtype = "auto"
                        quantization = (
                            precision
                            if precision in ["int8", "int4", "gptq", "awq"]
                            else None
                        )
                else:
                    model_path = model_name
                    dtype = "auto"
                    quantization = (
                        precision if precision in ["int8", "int4", "gptq"] else None
                    )
                    logger.warning(
                        f"Using base model with on-the-fly quantization: {precision}"
                    )

            llm = LLM(
                model=model_path,
                dtype=dtype,
                quantization=quantization,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_model_len,
                tensor_parallel_size=1,
                trust_remote_code=True,
            )

            logger.info("Model loaded successfully")
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU Memory allocated: {allocated:.2f}GB")

            return llm

        except Exception as e:
            logger.error(
                f"Failed to load {model_name} in {precision}: {e}", exc_info=True
            )
            return None

    def benchmark_latency_memory(
        self, llm: LLM, model_name: str, precision: str
    ) -> Dict:
        """Benchmark vLLM model for latency and memory."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
            top_p=1.0,
        )

        latencies = []
        peak_memories = []
        avg_memories = []
        tokens_generated = []

        total_iterations = (
            self.args.num_runs + self.args.num_warmup + self.args.batch_size - 1
        ) // self.args.batch_size

        logger.info("=" * 60)
        logger.info("LATENCY & MEMORY BENCHMARK")
        logger.info("=" * 60)
        logger.info(
            f"Warmup: {self.args.num_warmup} | Benchmark: {self.args.num_runs} | Batch: {self.args.batch_size}"
        )

        # Initialize NVML for memory tracking
        nvml_available = False
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                nvml_available = True
            except Exception as e:
                logger.warning(f"Could not initialize NVML: {e}")

        # Get baseline memory after model load (vLLM's KV cache allocation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            baseline_memory_mb = self.get_gpu_memory_mb()
            logger.info(
                f"Baseline GPU memory (model + KV cache): {baseline_memory_mb:.2f} MB"
            )
        else:
            baseline_memory_mb = 0

        iteration_count = 0
        pbar = tqdm(
            total=self.args.num_runs + self.args.num_warmup, desc="Latency/Memory"
        )

        for batch_idx in range(total_iterations):
            current_batch_size = min(
                self.args.batch_size,
                self.args.num_runs + self.args.num_warmup - iteration_count,
            )
            prompts = [self.args.prompt] * current_batch_size

            # Reset peak stats before each batch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            batch_latency = end_time - start_time

            # Measure memory: peak during generation and current allocation
            peak_mem = self.get_peak_gpu_memory_mb()
            avg_mem = self.get_gpu_memory_mb()

            per_request_latency = batch_latency / current_batch_size
            batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

            for i in range(current_batch_size):
                if iteration_count >= self.args.num_warmup:
                    latencies.append(per_request_latency)
                    # Use baseline memory if measurements are 0
                    peak_memories.append(
                        peak_mem if peak_mem > 0 else baseline_memory_mb
                    )
                    avg_memories.append(avg_mem if avg_mem > 0 else baseline_memory_mb)
                    tokens_generated.append(batch_tokens / current_batch_size)

                iteration_count += 1
                pbar.update(1)

                if iteration_count >= self.args.num_runs + self.args.num_warmup:
                    break

            if iteration_count >= self.args.num_runs + self.args.num_warmup:
                break

        pbar.close()

        # Cleanup NVML if initialized
        if nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

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
            "baseline_memory_mb": baseline_memory_mb,
        }

        logger.info(
            f"Latency: {results['mean_latency_s']:.4f}s | Memory: {results['mean_peak_mem_mb']:.2f}MB"
        )
        return results

    def benchmark_energy(self, llm: LLM, model_name: str, precision: str) -> Dict:
        """Measure energy consumption using PyNVML for accurate GPU power tracking."""
        logger.info("=" * 60)
        logger.info("ENERGY CONSUMPTION BENCHMARK (PyNVML)")
        logger.info("=" * 60)
        logger.info(
            f"Running {self.args.energy_sample_runs} inference samples with energy tracking..."
        )

        if not PYNVML_AVAILABLE:
            logger.error("PyNVML not available. Install with: pip install nvidia-ml-py")
            return {
                "energy_kwh": 0,
                "energy_joules": 0,
                "duration_seconds": 0,
                "avg_power_watts": 0,
                "min_power_watts": 0,
                "max_power_watts": 0,
                "std_power_watts": 0,
                "num_samples": 0,
                "energy_per_query_j": 0,
                "error": "PyNVML not available",
            }

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
            top_p=1.0,
        )

        test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main causes of climate change?",
            "Solve: If x + 5 = 12, what is x?",
            "Describe the process of photosynthesis.",
        ]

        try:
            # Initialize NVML
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            power_samples = []
            stop_monitoring = threading.Event()

            def monitor_power():
                """Background thread to monitor GPU power consumption."""
                while not stop_monitoring.is_set():
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_w = power_mw / 1000.0  # Convert milliwatts to watts
                        power_samples.append(power_w)
                    except Exception as e:
                        logger.debug(f"Power sampling error: {e}")
                    time.sleep(0.1)  # Sample every 100ms

            # Start power monitoring thread
            monitor_thread = threading.Thread(target=monitor_power, daemon=True)
            monitor_thread.start()

            start_time = time.time()

            # Run inference samples
            for i in tqdm(range(self.args.energy_sample_runs), desc="Energy tracking"):
                prompt = test_prompts[i % len(test_prompts)]
                prompts_batch = [prompt]
                llm.generate(prompts_batch, sampling_params)

            end_time = time.time()

            # Stop power monitoring
            stop_monitoring.set()
            monitor_thread.join(timeout=2.0)

            # Shutdown NVML
            pynvml.nvmlShutdown()

            # Calculate energy metrics
            duration = end_time - start_time

            if len(power_samples) == 0:
                raise Exception("No power samples collected")

            power_array = np.array(power_samples)
            avg_power = power_array.mean()
            energy_joules = avg_power * duration
            energy_kwh = energy_joules / 3600000

            results = {
                "energy_kwh": energy_kwh,
                "energy_joules": energy_joules,
                "duration_seconds": duration,
                "avg_power_watts": avg_power,
                "min_power_watts": power_array.min(),
                "max_power_watts": power_array.max(),
                "std_power_watts": power_array.std(),
                "num_samples": self.args.energy_sample_runs,
                "energy_per_query_j": energy_joules / self.args.energy_sample_runs,
                "power_samples_collected": len(power_samples),
            }

            logger.info(
                f"Energy: {results['energy_kwh'] * 1000:.4f} Wh | "
                f"Avg Power: {results['avg_power_watts']:.2f}W | "
                f"Range: {results['min_power_watts']:.1f}-{results['max_power_watts']:.1f}W"
            )
            return results

        except Exception as e:
            logger.error(f"Energy tracking failed: {e}", exc_info=True)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return {
                "energy_kwh": 0,
                "energy_joules": 0,
                "duration_seconds": 0,
                "avg_power_watts": 0,
                "min_power_watts": 0,
                "max_power_watts": 0,
                "std_power_watts": 0,
                "num_samples": 0,
                "energy_per_query_j": 0,
                "error": str(e),
            }

    def benchmark_accuracy(self, model_name: str, precision: str) -> Dict:
        """Evaluate model accuracy using lm-evaluation-harness with vLLM backend.

        Runs each requested task sequentially so we can apply per-task batch sizes
        to reduce OOM risk (e.g., a smaller batch for MMLU while keeping others larger).
        """
        logger.info("=" * 60)
        logger.info("ACCURACY EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Tasks: {', '.join(self.args.accuracy_tasks)}")
        logger.info(f"Few-shot: {self.args.num_fewshot}")
        if self.args.accuracy_limit:
            logger.info(f"Limit: {self.args.accuracy_limit} examples per task")

        try:
            if precision == "fp16":
                model_path = model_name
                model_lower = model_name.lower()

                # Model-specific dtype based on HuggingFace documentation
                # Use bfloat16 for: Llama, Mistral, Gemma, Phi, Qwen
                if any(
                    x in model_lower
                    for x in ["gemma", "llama", "mistral", "phi", "qwen"]
                ):
                    dtype = "bfloat16"
                else:
                    dtype = "float16"

                model_args = (
                    f"pretrained={model_path},dtype={dtype},"
                    f"gpu_memory_utilization={self.args.gpu_memory_utilization},"
                    f"trust_remote_code=True,max_model_len={self.args.max_model_len},"
                    f"tensor_parallel_size=1"
                )
            else:
                model_short_name = model_name.split("/")[-1]
                quantized_path = (
                    self.quantized_models_dir / f"{model_short_name}_{precision}"
                )

                if quantized_path.exists():
                    model_path = str(quantized_path)
                    logger.info(f"Using pre-quantized model: {model_path}")
                else:
                    model_path = model_name
                    logger.info(
                        f"Using base model with on-the-fly quantization: {precision}"
                    )

                model_args = (
                    f"pretrained={model_path},dtype=auto,"
                    f"gpu_memory_utilization={self.args.gpu_memory_utilization},"
                    f"trust_remote_code=True,max_model_len={self.args.max_model_len},"
                    f"tensor_parallel_size=1"
                )

                if not quantized_path.exists() and precision in [
                    "int8",
                    "int4",
                    "gptq",
                ]:
                    model_args += f",quantization={precision}"

            # Helper to resolve per-task batch size
            def resolve_batch_size(task: str) -> int:
                task = task.lower()
                if (
                    task == "hellaswag"
                    and self.args.accuracy_batch_size_hellaswag is not None
                ):
                    return self.args.accuracy_batch_size_hellaswag
                if task == "gsm8k" and self.args.accuracy_batch_size_gsm8k is not None:
                    return self.args.accuracy_batch_size_gsm8k
                if task == "mmlu" and self.args.accuracy_batch_size_mmlu is not None:
                    return self.args.accuracy_batch_size_mmlu
                return self.args.accuracy_batch_size

            accuracy_results = {}
            for task in self.args.accuracy_tasks:
                bs = resolve_batch_size(task)
                logger.info(f"Running accuracy task '{task}' with batch_size={bs}")

                results = evaluator.simple_evaluate(
                    model="vllm",
                    model_args=model_args,
                    tasks=[task],
                    num_fewshot=self.args.num_fewshot,
                    batch_size=bs,
                    limit=self.args.accuracy_limit,
                    log_samples=False,
                )

                # Extract the results for this single task
                for task_name, task_results in results["results"].items():
                    if "acc" in task_results:
                        accuracy_results[f"{task_name}_accuracy"] = task_results["acc"]
                    elif "acc_norm" in task_results:
                        accuracy_results[f"{task_name}_accuracy"] = task_results[
                            "acc_norm"
                        ]
                    elif "exact_match" in task_results:
                        accuracy_results[f"{task_name}_accuracy"] = task_results[
                            "exact_match"
                        ]
                    elif "pass@1" in task_results:
                        accuracy_results[f"{task_name}_accuracy"] = task_results[
                            "pass@1"
                        ]
                    else:
                        for key, value in task_results.items():
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                accuracy_results[f"{task_name}_accuracy"] = value
                                break

            logger.info("Accuracy results:")
            for task, acc in accuracy_results.items():
                logger.info(f"  {task}: {acc:.4f} ({acc * 100:.2f}%)")

            return accuracy_results

        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}", exc_info=True)
            return {f"{task}_accuracy": 0 for task in self.args.accuracy_tasks}

    def run_complete_benchmark(self, model_name: str, precision: str) -> Optional[Dict]:
        """Run complete benchmark: latency, memory, energy, and accuracy."""
        logger.info("#" * 70)
        logger.info(f"COMPLETE BENCHMARK: {model_name} ({precision.upper()})")
        logger.info("#" * 70)

        # Get model size information
        model_size_info = self.get_model_size(model_name)
        logger.info(
            f"Model size: {model_size_info['num_parameters_b']:.2f}B parameters ({model_size_info['size_gb_fp16']:.2f} GB in FP16)"
        )

        results = {
            "model": model_name,
            "precision": precision,
            "quantization_scheme": (
                "FP16"
                if precision == "fp16"
                else self.quantization_configs.get(precision, {}).get(
                    "scheme_name", precision.upper()
                )
            ),
            "timestamp": datetime.now().isoformat(),
            "num_parameters": model_size_info["num_parameters"],
            "num_parameters_b": model_size_info["num_parameters_b"],
            "size_gb_fp16": model_size_info["size_gb_fp16"],
        }

        llm = self.setup_vllm_model(model_name, precision)
        if llm is None:
            logger.error("Model setup failed, skipping...")
            return None

        try:
            # 1. Latency & Memory (if enabled)
            if self.args.enable_latency_memory:
                latency_memory_results = self.benchmark_latency_memory(
                    llm, model_name, precision
                )
                results.update(latency_memory_results)

            # 2. Energy (if enabled)
            if self.args.enable_energy_tracking:
                energy_results = self.benchmark_energy(llm, model_name, precision)
                results.update(energy_results)

            # Clean up vLLM model before accuracy eval
            del llm
            self.clear_cache()
            time.sleep(2)

            # 3. Accuracy (if enabled)
            if self.args.run_accuracy:
                accuracy_results = self.benchmark_accuracy(model_name, precision)
                results.update(accuracy_results)
            else:
                # Populate accuracy columns with NaN when accuracy is not run
                accuracy_results = {
                    f"{task}_accuracy": float("nan")
                    for task in self.args.accuracy_tasks
                }
                results.update(accuracy_results)

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return None
        finally:
            self.clear_cache()

    def initialize_results_csv(self):
        """Initialize results CSV with all columns."""
        base_columns = [
            "model",
            "precision",
            "quantization_scheme",
            "timestamp",
            "num_parameters",
            "num_parameters_b",
            "size_gb_fp16",
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
            "min_power_watts",
            "max_power_watts",
            "std_power_watts",
            "energy_per_query_j",
        ]

        accuracy_columns = [f"{task}_accuracy" for task in self.args.accuracy_tasks]

        all_columns = base_columns + energy_columns + accuracy_columns

        if not self.results_csv.exists():
            pd.DataFrame(columns=all_columns).to_csv(self.results_csv, index=False)
            logger.info(f"Created results CSV: {self.results_csv}")
        else:
            logger.info(f"Results CSV exists: {self.results_csv}")

    def run_all_benchmarks(self):
        """Run benchmarks for all model-precision combinations."""
        logger.info("#" * 70)
        logger.info("SLiM-Eval: COMPLETE BENCHMARK SUITE")
        logger.info("#" * 70)
        logger.info(f"Models: {len(self.args.models)}")
        logger.info(f"Precision: {self.args.precision}")
        logger.info(f"Total configs: {len(self.args.models)}")
        logger.info("Metrics: Latency, Memory, Energy, Accuracy")
        logger.info(f"Output: {self.results_csv}")
        logger.info(f"Estimated time: ~{len(self.args.models) * 30} minutes")

        self.initialize_results_csv()

        all_results = []

        for model_name in self.args.models:
            precision = self.args.precision
            config_id = f"{model_name.split('/')[-1]}_{precision}"

            results = self.run_complete_benchmark(model_name, precision)

            if results:
                all_results.append(results)

                # Read the existing CSV to get the exact column order
                existing_df = pd.read_csv(self.results_csv)
                columns = existing_df.columns.tolist()

                # Create DataFrame with results in the correct column order
                results_df = pd.DataFrame([results], columns=columns)

                # Append to CSV
                results_df.to_csv(self.results_csv, mode="a", header=False, index=False)
                logger.info(f"Results saved for {config_id}")

            self.clear_cache()
            time.sleep(5)

        logger.info("#" * 70)
        logger.info("ALL BENCHMARKS COMPLETE!")
        logger.info("#" * 70)

        return all_results

    def analyze_results(self):
        """Analyze and visualize results."""
        if not self.results_csv.exists():
            logger.warning("No results file found")
            return

        results_df = pd.read_csv(self.results_csv)

        logger.info("#" * 70)
        logger.info("COMPLETE RESULTS SUMMARY")
        logger.info("#" * 70)

        # Display key metrics
        display_columns = [
            "model",
            "num_parameters_b",
            "precision",
            "mean_latency_s",
            "mean_peak_mem_mb",
            "energy_kwh",
            "avg_power_watts",
        ] + [col for col in results_df.columns if "accuracy" in col]

        logger.info("\n" + results_df[display_columns].round(4).to_string(index=False))

        # Calculate efficiency metrics
        self._analyze_quantization_impact(results_df)

        # Generate visualizations
        self._generate_visualizations(results_df)

        # Generate summary statistics
        self._generate_summary_statistics(results_df)

        # Generate paper-ready table
        self._generate_paper_table(results_df)

        # Generate executive summary
        self._generate_executive_summary(results_df)

        # Export JSON
        self._export_json(results_df)

    def _analyze_quantization_impact(self, results_df):
        """Analyze quantization impact on metrics."""
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
                        "model": str(model).split("/")[-1],
                        "precision": row["precision"],
                        "speedup": fp16_latency / row["mean_latency_s"],
                        "memory_reduction_pct": (
                            1 - row["mean_peak_mem_mb"] / fp16_memory
                        )
                        * 100,
                    }

                    if "energy_joules" in row and fp16_energy > 0:
                        analysis_row["energy_reduction_pct"] = (
                            1 - row["energy_joules"] / fp16_energy
                        ) * 100

                    for task in self.args.accuracy_tasks:
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
            logger.info("\n" + "#" * 70)
            logger.info("QUANTIZATION IMPACT ANALYSIS")
            logger.info("#" * 70)
            logger.info("\n" + analysis_df.round(3).to_string(index=False))

            analysis_path = self.output_dir / "quantization_impact.csv"
            analysis_df.to_csv(analysis_path, index=False)
            logger.info(f"Analysis saved: {analysis_path}")

    def _generate_visualizations(self, results_df):
        """Generate visualization plots."""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Latency vs Memory
        ax1 = axes[0, 0]
        for precision in results_df["precision"].unique():
            data = results_df[results_df["precision"] == precision]
            ax1.scatter(
                data["mean_latency_s"],
                data["mean_peak_mem_mb"],
                label=str(precision).upper(),
                s=150,
                alpha=0.7,
            )
        ax1.set_xlabel("Latency (seconds)")
        ax1.set_ylabel("Peak Memory (MB)")
        ax1.set_title("Latency vs Memory")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Energy vs Accuracy
        if "energy_kwh" in results_df.columns:
            ax2 = axes[0, 1]
            acc_col = [col for col in results_df.columns if "accuracy" in col]
            if acc_col:
                acc_col = acc_col[0]
                for precision in results_df["precision"].unique():
                    data = results_df[results_df["precision"] == precision]
                    ax2.scatter(
                        data["energy_kwh"],
                        data[acc_col],
                        label=str(precision).upper(),
                        s=150,
                        alpha=0.7,
                    )
                ax2.set_xlabel("Energy (kWh)")
                ax2.set_ylabel("Accuracy")
                ax2.set_title("Energy vs Accuracy Trade-off")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # Plot 3: Throughput comparison
        ax3 = axes[1, 0]
        models_short = results_df["model"].astype(str).str.split("/").str[-1]
        x_positions = range(len(results_df))
        bar_width = 0.25

        precisions = results_df["precision"].unique()
        for i, precision in enumerate(precisions):
            data = results_df[results_df["precision"] == precision]
            indices = data.index
            positions = [
                x_positions[idx] + i * bar_width for idx in range(len(indices))
            ]
            ax3.bar(
                positions,
                data["tokens_per_second"],
                width=bar_width,
                label=str(precision).upper(),
                alpha=0.7,
            )

        ax3.set_xlabel("Model")
        ax3.set_ylabel("Tokens/Second")
        ax3.set_title("Throughput by Model and Precision")
        ax3.set_xticks([p + bar_width for p in x_positions])
        ax3.set_xticklabels(
            [models_short[i] for i in range(len(models_short))], rotation=45, ha="right"
        )
        ax3.legend()

        # Plot 4: Accuracy comparison
        ax4 = axes[1, 1]
        accuracy_cols = [col for col in results_df.columns if col.endswith("_accuracy")]
        # Only plot if we have accuracy columns with valid non-NaN data
        has_valid_accuracy = (
            accuracy_cols
            and not results_df[accuracy_cols].isna().all().all()
            and results_df[accuracy_cols].max().max() > 0
        )
        if has_valid_accuracy:
            results_df["avg_accuracy"] = results_df[accuracy_cols].mean(axis=1)
            for i, precision in enumerate(precisions):
                data = results_df[results_df["precision"] == precision]
                indices = data.index
                positions = [
                    x_positions[idx] + i * bar_width for idx in range(len(indices))
                ]
                ax4.bar(
                    positions,
                    data["avg_accuracy"],
                    width=bar_width,
                    label=str(precision).upper(),
                    alpha=0.7,
                )
            ax4.set_xlabel("Model")
            ax4.set_ylabel("Average Accuracy")
            ax4.set_title("Average Accuracy Across Tasks")
            ax4.set_xticks([p + bar_width for p in x_positions])
            ax4.set_xticklabels(
                [models_short[i] for i in range(len(models_short))],
                rotation=45,
                ha="right",
            )
            ax4.legend()
        else:
            # If no accuracy data, hide the plot
            ax4.text(
                0.5, 0.5, "No Accuracy Data", ha="center", va="center", fontsize=14
            )
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()
        plot_path = self.output_dir / "pareto_frontiers.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Visualizations saved: {plot_path}")

    def _generate_summary_statistics(self, results_df):
        """Generate summary statistics."""
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

        logger.info("\n" + "#" * 70)
        logger.info("SUMMARY STATISTICS BY PRECISION")
        logger.info("#" * 70)
        logger.info("\n" + str(summary_stats))

        summary_path = self.output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_path)
        logger.info(f"Summary saved: {summary_path}")

    def _generate_paper_table(self, results_df):
        """Generate paper-ready table."""
        paper_columns = [
            "model",
            "num_parameters_b",
            "precision",
            "mean_latency_s",
            "mean_peak_mem_mb",
            "tokens_per_second",
        ]

        if "energy_kwh" in results_df.columns:
            paper_columns.append("energy_kwh")

        accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
        if accuracy_cols:
            paper_columns.extend(accuracy_cols[:3])

        paper_table = results_df[paper_columns].copy()
        paper_table["model"] = paper_table["model"].astype(str).str.split("/").str[-1]

        rename_map = {
            "num_parameters_b": "Params (B)",
            "mean_latency_s": "Latency (s)",
            "mean_peak_mem_mb": "Memory (MB)",
            "tokens_per_second": "Tokens/s",
            "energy_kwh": "Energy (kWh)",
        }
        paper_table = paper_table.rename(columns=rename_map)
        paper_table = paper_table.round(4)

        logger.info("\n" + "#" * 70)
        logger.info("PAPER-READY RESULTS TABLE")
        logger.info("#" * 70)
        logger.info("\n" + paper_table.to_string(index=False))

        paper_csv_path = self.output_dir / "paper_results.csv"
        paper_table.to_csv(paper_csv_path, index=False)

        latex_table = paper_table.to_latex(index=False, float_format="%.4f")
        latex_path = self.output_dir / "paper_results.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)

        logger.info(f"Paper table saved: {paper_csv_path}")
        logger.info(f"LaTeX table saved: {latex_path}")

    def _generate_executive_summary(self, results_df):
        """Generate executive summary report."""
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

        if len(results_df) > 0:
            # 1. Fastest model (guard for NaNs)
            if "mean_latency_s" in results_df.columns:
                lat_series = results_df["mean_latency_s"].dropna()
                if not lat_series.empty:
                    fastest_idx = lat_series.idxmin()
                    fastest = results_df.loc[fastest_idx]
                    report_lines.append("\n1. FASTEST MODEL:")
                    report_lines.append(
                        f"   {str(fastest['model']).split('/')[-1]} ({fastest['precision']})"
                    )
                    report_lines.append(f"   Latency: {fastest['mean_latency_s']:.4f}s")

            # 2. Most memory efficient (guard for NaNs)
            if "mean_peak_mem_mb" in results_df.columns:
                mem_series = results_df["mean_peak_mem_mb"].dropna()
                if not mem_series.empty:
                    mem_idx = mem_series.idxmin()
                    mem_efficient = results_df.loc[mem_idx]
                    report_lines.append("\n2. MOST MEMORY EFFICIENT:")
                    report_lines.append(
                        f"   {str(mem_efficient['model']).split('/')[-1]} ({mem_efficient['precision']})"
                    )
                    report_lines.append(
                        f"   Memory: {mem_efficient['mean_peak_mem_mb']:.2f} MB"
                    )

            # 3. Highest throughput (guard for NaNs)
            if "tokens_per_second" in results_df.columns:
                tps_series = results_df["tokens_per_second"].dropna()
                if not tps_series.empty:
                    tps_idx = tps_series.idxmax()
                    highest_throughput = results_df.loc[tps_idx]
                    report_lines.append("\n3. HIGHEST THROUGHPUT:")
                    report_lines.append(
                        f"   {str(highest_throughput['model']).split('/')[-1]} ({highest_throughput['precision']})"
                    )
                    report_lines.append(
                        f"   Throughput: {highest_throughput['tokens_per_second']:.2f} tokens/s"
                    )

            accuracy_cols = [col for col in results_df.columns if "accuracy" in col]
            if accuracy_cols:
                # Compute average accuracy while ignoring NaNs
                avg_acc_series = results_df[accuracy_cols].mean(axis=1, skipna=True)
                # Filter out NaNs and non-positive
                valid_avg = avg_acc_series.dropna()
                if not valid_avg.empty and valid_avg.max() > 0:
                    best_idx = valid_avg.idxmax()
                    best_accuracy = results_df.loc[best_idx]
                    report_lines.append("\n4. BEST AVERAGE ACCURACY:")
                    report_lines.append(
                        f"   {str(best_accuracy['model']).split('/')[-1]} ({best_accuracy['precision']})"
                    )
                    report_lines.append(
                        f"   Avg Accuracy: {valid_avg.loc[best_idx]:.4f}"
                    )

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)
        logger.info("\n" + report_text)

        report_path = self.output_dir / "executive_summary.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        logger.info(f"Executive summary saved: {report_path}")

    def _export_json(self, results_df):
        """Export results as JSON."""
        json_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models": self.args.models,
                "precision": self.args.precision,
                "accuracy_tasks": self.args.accuracy_tasks,
                "num_runs": self.args.num_runs,
                "batch_size": self.args.batch_size,
            },
            "results": results_df.to_dict(orient="records"),
        }

        json_path = self.output_dir / "complete_results.json"
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)

        logger.info(f"JSON results saved: {json_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SLiM-Eval: Small Language Model Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.2-3B"],
        help="List of model names to evaluate",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "int4", "gptq"],
        help=(
            "Precision to test: fp16 (baseline), int8/int4 (SmoothQuant+GPTQ), gptq (GPTQ-only)"
        ),
    )

    # Benchmark configuration
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1000,
        help="Number of latency benchmark runs",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain one interesting fact about large language models.",
        help="Prompt for benchmarking",
    )

    # Accuracy configuration (controlled via --benchmarks)
    parser.add_argument(
        "--accuracy-tasks",
        nargs="+",
        default=["mmlu", "gsm8k", "hellaswag"],
        help="Accuracy evaluation tasks",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--accuracy-limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing)",
    )
    parser.add_argument(
        "--accuracy-batch-size",
        type=int,
        default=32,
        help="Batch size used by lm-eval during accuracy evaluation (reduce to avoid GPU OOM)",
    )
    parser.add_argument(
        "--accuracy-batch-size-hellaswag",
        type=int,
        default=None,
        help="Override batch size for Hellaswag only (falls back to --accuracy-batch-size)",
    )
    parser.add_argument(
        "--accuracy-batch-size-gsm8k",
        type=int,
        default=None,
        help="Override batch size for GSM8K only (falls back to --accuracy-batch-size)",
    )
    parser.add_argument(
        "--accuracy-batch-size-mmlu",
        type=int,
        default=None,
        help="Override batch size for MMLU only (falls back to --accuracy-batch-size)",
    )

    # Energy configuration (controlled via --benchmarks)
    parser.add_argument(
        "--energy-sample-runs",
        type=int,
        default=100,
        help="Number of runs for energy measurement",
    )

    # Unified benchmark selection
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["latency_memory", "energy", "accuracy"],
        default=None,
        help=(
            "Select which benchmarks to run. Choices: latency_memory, energy, accuracy. "
            "If provided, overrides legacy flags like --run-accuracy and --enable-energy-tracking."
        ),
    )

    # Quantization configuration
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset for quantization",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default="train_sft",
        help="Calibration dataset split",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration",
    )

    # Model loading configuration
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model length for vLLM",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="slim_eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quantized-models-dir",
        type=str,
        default="quantized_models",
        help="Directory for quantized models",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "benchmark", "analyze"],
        help="Run mode: 'all' (benchmark + analyze), 'benchmark' (only run benchmarks), 'analyze' (only analyze existing results)",
    )

    args = parser.parse_args()

    # Unified control: derive booleans from --benchmarks exclusively
    selected = set(args.benchmarks or [])
    args.enable_latency_memory = "latency_memory" in selected
    args.enable_energy_tracking = "energy" in selected
    args.run_accuracy = "accuracy" in selected

    return args


def main():
    """Main entry point."""
    args = parse_args()

    evaluator = SLiMEvaluator(args)

    try:
        if args.mode in ["all", "benchmark"]:
            evaluator.run_all_benchmarks()

        if args.mode in ["all", "analyze"]:
            evaluator.analyze_results()

        logger.info("SLiM-Eval complete!")
    finally:
        # Ensure proper cleanup of distributed resources
        evaluator.clear_cache()


if __name__ == "__main__":
    main()
