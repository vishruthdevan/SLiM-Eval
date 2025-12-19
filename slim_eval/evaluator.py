import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

# Optional heavy dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from vllm import LLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    if TYPE_CHECKING:
        from vllm import LLM

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .analysis import ResultsAnalyzer

# Import benchmarks and utilities only if dependencies are available
if TORCH_AVAILABLE and VLLM_AVAILABLE:
    from .benchmarks.accuracy import AccuracyBenchmark
    from .benchmarks.energy import EnergyBenchmark
    from .benchmarks.performance import PerformanceBenchmark
    from .quantization import QuantizationManager
    from .utils import clear_cache, get_model_size, get_quantized_model_size

logger = logging.getLogger(__name__)


class SLiMEvaluator:
    """Main evaluator orchestrating quantization and benchmarking."""

    def __init__(self, args):
        """Initialize SLiM-Eval.

        Args:
            args: Arguments object containing all configuration.
        """
        self.args = args
        self.wandb_run = None

        # Initialize wandb if enabled
        if getattr(self.args, "wandb_enabled", False):
            if WANDB_AVAILABLE:
                self._init_wandb()
            else:
                logger.warning("wandb not available, skipping initialization")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.quantized_models_dir = Path(args.quantized_models_dir)
        self.quantized_models_dir.mkdir(exist_ok=True)

        # Check if heavy dependencies are available for benchmarking
        self.benchmarking_available = TORCH_AVAILABLE and VLLM_AVAILABLE

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

        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Select GPU
            try:
                import os

                torch.cuda.set_device(self.args.gpu_index)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_index)
            except Exception as e:
                logger.warning(f"Failed to set GPU index {self.args.gpu_index}: {e}")

            logger.info(f"GPU: {torch.cuda.get_device_name(self.args.gpu_index)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(self.args.gpu_index).total_memory / 1024**3:.2f} GB"
            )
        elif TORCH_AVAILABLE:
            logger.warning("CUDA not available")
        else:
            logger.info("PyTorch not available (analysis mode only)")

        # Initialize benchmark components (only if dependencies available)
        if self.benchmarking_available:
            self.quantization_manager = QuantizationManager(args)
            self.performance_benchmark = PerformanceBenchmark(args)
            self.energy_benchmark = EnergyBenchmark(args)
            self.accuracy_benchmark = AccuracyBenchmark(
                args, self.quantized_models_dir, self.output_dir
            )
        else:
            self.quantization_manager = None
            self.performance_benchmark = None
            self.energy_benchmark = None
            self.accuracy_benchmark = None
            if hasattr(args, "models") and args.models:
                logger.warning(
                    "Heavy dependencies (torch, vllm, llmcompressor) not available. "
                    "Only 'analyze' command will work. "
                    "To run benchmarks, install with: pip install -e '.[full]'"
                )

        # Analyzer is always available (only needs pandas, matplotlib)
        # Use input_dir if provided (for analyze command), otherwise use output_dir
        input_dir = Path(getattr(args, "input_dir", args.output_dir))
        self.analyzer = ResultsAnalyzer(input_dir, self.output_dir, args.accuracy_tasks)

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            # Set API key if provided
            if hasattr(self.args, "wandb_api_key") and self.args.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.args.wandb_api_key

            # Auto-generate model name for grouping
            model_name = (
                self.args.models[0].split("/")[-1] if self.args.models else "unknown"
            )

            # Generate run name if not provided
            if hasattr(self.args, "wandb_run_name") and self.args.wandb_run_name:
                run_name = self.args.wandb_run_name
            else:
                # Auto-generate: model_precision (e.g., "Llama-3.2-3B_int8")
                run_name = f"{model_name}_{self.args.precision}"

            # Prepare config dictionary
            config = {
                "models": self.args.models,
                "precision": self.args.precision,
                "batch_size": self.args.batch_size,
                "max_new_tokens": self.args.max_new_tokens,
                "num_runs": self.args.num_runs,
                "num_warmup": self.args.num_warmup,
                "gpu_memory_utilization": self.args.gpu_memory_utilization,
                "max_model_len": self.args.max_model_len,
                "enable_performance_tracking": self.args.enable_performance_tracking,
                "enable_energy_tracking": self.args.enable_energy_tracking,
                "enable_accuracy_tracking": self.args.enable_accuracy_tracking,
            }

            # Add accuracy-specific config if enabled
            if self.args.enable_accuracy_tracking:
                config.update(
                    {
                        "accuracy_tasks": self.args.accuracy_tasks,
                        "num_fewshot": self.args.num_fewshot,
                        "accuracy_batch_size": self.args.accuracy_batch_size,
                    }
                )

            # Add quantization config if not fp16
            if self.args.precision != "fp16":
                config.update(
                    {
                        "calibration_dataset": self.args.calibration_dataset,
                        "num_calibration_samples": self.args.num_calibration_samples,
                        "max_sequence_length": self.args.max_sequence_length,
                    }
                )

            # Initialize wandb
            self.wandb_run = wandb.init(
                project=self.args.wandb_project,
                name=run_name,
                config=config,
                tags=[self.args.precision, "quantization-benchmark"],
                group=model_name.split("/")[-1],  # groups runs by model
            )

            logger.info(f"Weights & Biases initialized: {run_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.wandb_run = None

    def _log_to_wandb(self, results: Dict[str, Any], model_name: str, precision: str):
        """Log results to Weights & Biases.

        Args:
            results: Results dictionary to log (flat structure).
            model_name: Name of the model.
            precision: Precision mode used.
        """
        try:
            # Prepare metrics for logging
            wandb_metrics = {
                "model": model_name,
                "precision": precision,
                "quantization_scheme": results.get("quantization_scheme", "N/A"),
            }

            # Add model info (if available)
            if "num_parameters_b" in results:
                wandb_metrics["model/num_parameters_b"] = results["num_parameters_b"]
            if "model_size_gb" in results:
                wandb_metrics["model/model_size_gb"] = results["model_size_gb"]
            if "num_parameters" in results:
                wandb_metrics["model/num_parameters"] = results["num_parameters"]

            # Add performance metrics (if available)
            performance_keys = {
                "mean_latency_s": "latency/mean_s",
                "median_latency_s": "latency/median_s",
                "p95_latency_s": "latency/p95_s",
                "p99_latency_s": "latency/p99_s",
                "std_latency_s": "latency/std_s",
                "mean_peak_mem_mb": "memory/peak_mb",
                "mean_avg_mem_mb": "memory/avg_mb",
                "baseline_memory_mb": "memory/baseline_mb",
                "tokens_per_second": "throughput/tokens_per_sec",
            }

            for result_key, wandb_key in performance_keys.items():
                if result_key in results:
                    wandb_metrics[wandb_key] = results[result_key]

            # Add energy metrics (if available)
            energy_keys = {
                "energy_kwh": "energy/kwh",
                "energy_joules": "energy/joules",
                "duration_seconds": "energy/duration_s",
                "avg_power_watts": "energy/avg_power_watts",
                "min_power_watts": "energy/min_power_watts",
                "max_power_watts": "energy/max_power_watts",
                "std_power_watts": "energy/std_power_watts",
                "energy_per_query_j": "energy/per_query_j",
            }

            for result_key, wandb_key in energy_keys.items():
                if result_key in results:
                    wandb_metrics[wandb_key] = results[result_key]

            # Add accuracy metrics (if available)
            for task in self.args.accuracy_tasks:
                accuracy_key = f"{task}_accuracy"
                if accuracy_key in results:
                    value = results[accuracy_key]
                    wandb_metrics[f"accuracy/{task}"] = value
                    wandb_metrics[f"accuracy/{task}_pct"] = value * 100

            # Log all metrics as both logs (for charts) and summary (for comparison)
            wandb.log(wandb_metrics)

            # Also set as summary for better comparison view
            for key, value in wandb_metrics.items():
                wandb.summary[key] = value

            # Create a summary table for better visualization
            summary_data = []

            # Add performance row if data exists
            if "mean_latency_s" in results:
                summary_data.append(
                    [
                        "Latency (mean)",
                        f"{results['mean_latency_s']:.4f} s",
                        f"{results.get('tokens_per_second', 0):.2f} tok/s",
                    ]
                )

            if "mean_peak_mem_mb" in results:
                summary_data.append(
                    [
                        "Memory (peak)",
                        f"{results['mean_peak_mem_mb']:.2f} MB",
                        f"{results.get('baseline_memory_mb', 0):.2f} MB baseline",
                    ]
                )

            # Add energy row if data exists
            if "energy_kwh" in results:
                summary_data.append(
                    [
                        "Energy",
                        f"{results['energy_kwh'] * 1000:.4f} Wh",
                        f"{results['avg_power_watts']:.2f} W avg",
                    ]
                )

            # Add accuracy rows
            for task in self.args.accuracy_tasks:
                accuracy_key = f"{task}_accuracy"
                if accuracy_key in results:
                    summary_data.append(
                        [
                            f"Accuracy ({task})",
                            f"{results[accuracy_key]:.4f}",
                            f"{results[accuracy_key] * 100:.2f}%",
                        ]
                    )

            # Log summary table
            if summary_data:
                summary_table = wandb.Table(
                    columns=["Metric", "Value", "Additional Info"], data=summary_data
                )
                wandb.log({"results_summary": summary_table})

            logger.info(f"Logged {len(wandb_metrics)} metrics to W&B")

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def setup_vllm_model(
        self, model_name: str, precision: str, use_quantized_dir: bool = True
    ) -> Optional["LLM"]:
        """Setup and load a vLLM model.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Precision mode (fp16, int8, int4).
            use_quantized_dir: Whether to use pre-quantized models from disk.

        Returns:
            Loaded vLLM instance or None if loading failed.
        """
        if not self.benchmarking_available:
            logger.error("Cannot setup vLLM model: Heavy dependencies not available.")
            return None

        clear_cache()
        try:
            logger.info("=" * 60)
            logger.info(f"Loading {model_name} in {precision.upper()} precision...")
            logger.info("=" * 60)

            model_short_name = model_name.split("/")[-1]

            if precision == "fp16":
                # Check if we have a local copy in the quantized models directory
                fp16_path = self.quantized_models_dir / f"{model_short_name}_fp16"

                if fp16_path.exists() and (fp16_path / "config.json").exists():
                    model_path = str(fp16_path)
                    logger.info(f"Using local FP16 model from: {fp16_path}")
                else:
                    # Download and save the FP16 model locally
                    logger.info(f"Downloading and saving FP16 model to: {fp16_path}")
                    self._save_fp16_model(model_name, fp16_path)
                    model_path = str(fp16_path)

                model_lower = model_name.lower()

                # Model-specific dtype based on HuggingFace documentation
                # Llama 3.2: Uses bfloat16 (torch.bfloat16 in examples)
                # Mistral 7B: Uses bfloat16 (torch.bfloat16 in examples)
                # Gemma 2/3: Uses bfloat16 (native weights in bfloat16)
                # Phi-3: Uses "auto" which resolves to bfloat16
                # Qwen2.5: Uses "auto" which resolves to bfloat16

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
                quantized_path = (
                    self.quantized_models_dir / f"{model_short_name}_{precision}"
                )

                if use_quantized_dir and quantized_path.exists():
                    model_path = str(quantized_path)
                    dtype = "auto"
                    quantization = None
                    quant_config = self.quantization_manager.get_quantization_config(
                        precision
                    )
                    logger.info(f"Using pre-quantized model from: {quantized_path}")
                    logger.info(f"  Method: {quant_config.get('method', 'unknown')}")
                elif use_quantized_dir:
                    logger.info(f"Pre-quantized model not found at: {quantized_path}")
                    logger.info("Running quantization with llm-compressor...")
                    self.quantization_manager.quantize_model(
                        model_name, precision, quantized_path
                    )

                    # Clear GPU memory after quantization
                    logger.info("Clearing GPU memory after quantization...")
                    clear_cache()
                    time.sleep(5)  # Give time for memory to be fully released

                    # Check if quantization actually succeeded by verifying config.json exists
                    if (
                        quantized_path.exists()
                        and (quantized_path / "config.json").exists()
                    ):
                        model_path = str(quantized_path)
                        dtype = "auto"
                        quantization = None
                        logger.info(
                            f"Using newly quantized model from: {quantized_path}"
                        )
                    else:
                        logger.error(
                            f"Quantization failed for {model_name} with precision {precision}"
                        )
                        logger.error(
                            "Quantized model directory does not exist or is incomplete."
                        )
                        raise RuntimeError(
                            f"Quantization failed: Could not create valid quantized model at {quantized_path}"
                        )
                else:
                    logger.error(
                        f"use_quantized_dir=False is not supported. Quantization is required for {precision} precision."
                    )
                    raise RuntimeError(
                        f"Cannot load model in {precision} precision without quantization"
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
                # Check memory across all GPUs
                total_allocated = 0
                for device_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                    total_allocated += allocated
                    if allocated > 0:
                        logger.info(
                            f"GPU {device_id} Memory allocated: {allocated:.2f}GB"
                        )

                if total_allocated > 0:
                    logger.info(f"Total GPU Memory allocated: {total_allocated:.2f}GB")
                else:
                    # Fallback: show reserved memory (which vLLM uses)
                    total_reserved = sum(
                        torch.cuda.memory_reserved(i) / 1024**3
                        for i in range(torch.cuda.device_count())
                    )
                    if total_reserved > 0:
                        logger.info(
                            f"Total GPU Memory reserved: {total_reserved:.2f}GB"
                        )

            return llm
        except Exception as e:
            logger.error(
                f"Failed to load {model_name} in {precision}: {e}", exc_info=True
            )
            return None

    def _save_fp16_model(self, model_name: str, output_dir: Path) -> None:
        """Download and save an FP16 model locally.

        Args:
            model_name: HuggingFace model name or local path.
            output_dir: Directory to save the model.
        """
        import gc

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("=" * 60)
        logger.info(f"Downloading {model_name} for local storage")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Loading model and tokenizer from HuggingFace...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            logger.info(f"Saving model to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            logger.info(f"FP16 model saved successfully: {output_dir}")

            # Cleanup
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to save FP16 model: {e}", exc_info=True)
            raise

    def run_complete_benchmark(self, model_name: str, precision: str) -> Optional[Dict]:
        """Run complete benchmark suite for a model configuration.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Precision mode to evaluate.

        Returns:
            Dictionary of results or None if benchmark failed.
        """
        if not self.benchmarking_available:
            logger.error("Cannot run benchmark: Heavy dependencies not available.")
            return None

        logger.info("#" * 70)
        logger.info(f"COMPLETE BENCHMARK: {model_name} ({precision.upper()})")
        logger.info("#" * 70)

        model_size_info = get_model_size(model_name)
        model_short_name = model_name.split("/")[-1]

        logger.info(f"Model: {model_size_info['num_parameters_b']:.2f}B parameters")

        # Get quantization scheme name
        model_path = self.quantized_models_dir / f"{model_short_name}_{precision}"

        if precision == "fp16":
            scheme_name = "FP16"
        else:
            quant_config = self.quantization_manager.get_quantization_config(precision)
            scheme_name = quant_config.get("scheme_name", precision.upper())

        results = {
            "model": model_name,
            "precision": precision,
            "quantization_scheme": scheme_name,
            "timestamp": datetime.now().isoformat(),
            "num_parameters": model_size_info["num_parameters"],
            "num_parameters_b": model_size_info["num_parameters_b"],
            "model_size_gb": 0.0,  # Will be calculated after model is downloaded
        }

        llm = self.setup_vllm_model(model_name, precision)
        if llm is None:
            logger.error("Model setup failed, skipping...")
            return None

        # Calculate actual model size from disk after setup (model is now downloaded)
        if model_path.exists():
            actual_size = get_quantized_model_size(str(model_path))
            if actual_size > 0:
                results["model_size_gb"] = actual_size
                logger.info(f"Actual model size from disk: {actual_size:.2f} GB")

        try:
            # Run performance benchmark (latency & memory)
            if self.args.enable_performance_tracking:
                results.update(
                    self.performance_benchmark.run(llm, model_name, precision)
                )
                # Save performance results immediately
                self.save_results_to_json(model_name, precision, results)

            # Run energy benchmark
            if self.args.enable_energy_tracking:
                results.update(self.energy_benchmark.run(llm, model_name, precision))
                # Save energy results immediately
                self.save_results_to_json(model_name, precision, results)

            # Clean up vLLM instance before accuracy testing
            del llm
            clear_cache()
            time.sleep(2)

            # Run accuracy benchmark (creates its own vLLM instance)
            if self.args.enable_accuracy_tracking:
                # Prepare metadata for individual task saves
                task_metadata = {
                    "model": model_name,
                    "precision": precision,
                    "quantization_scheme": results.get("quantization_scheme"),
                    "timestamp": results.get("timestamp"),
                    "num_parameters": results.get("num_parameters"),
                    "num_parameters_b": results.get("num_parameters_b"),
                    "model_size_gb": results.get("model_size_gb"),
                }
                results.update(
                    self.accuracy_benchmark.run(
                        None, model_name, precision, task_metadata
                    )
                )
                # Save accuracy results immediately (this will now be redundant as tasks save individually)
                self.save_results_to_json(model_name, precision, results)
            else:
                results.update(
                    {
                        f"{task}_accuracy": float("nan")
                        for task in self.args.accuracy_tasks
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return None
        finally:
            clear_cache()

    def save_results_to_json(self, model_name: str, precision: str, results: Dict):
        """Save benchmark results as separate JSON files in model-specific directory.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Precision mode.
            results: Dictionary of results.
        """
        # Create model_precision directory
        model_short_name = model_name.split("/")[-1]
        model_dir = self.output_dir / f"{model_short_name}_{precision}"
        model_dir.mkdir(exist_ok=True)

        # Common metadata for all files
        metadata = {
            "model": model_name,
            "precision": precision,
            "quantization_scheme": results.get("quantization_scheme"),
            "timestamp": results.get("timestamp"),
            "num_parameters": results.get("num_parameters"),
            "num_parameters_b": results.get("num_parameters_b"),
            "model_size_gb": results.get("model_size_gb"),
        }

        # Save performance.json (only if performance data exists)
        performance_keys = [
            "mean_latency_s",
            "median_latency_s",
            "p95_latency_s",
            "p99_latency_s",
            "std_latency_s",
            "mean_peak_mem_mb",
            "mean_avg_mem_mb",
            "tokens_per_second",
        ]
        has_performance_data = any(key in results for key in performance_keys)

        if has_performance_data:
            performance_data = metadata.copy()
            for key in performance_keys:
                if key in results:
                    performance_data[key] = results[key]

            performance_file = model_dir / "performance.json"
            with open(performance_file, "w") as f:
                json.dump(performance_data, f, indent=2)
            logger.info(f"Saved performance results: {performance_file}")

        # Save energy.json (only if energy data exists)
        energy_keys = [
            "energy_kwh",
            "energy_joules",
            "duration_seconds",
            "avg_power_watts",
            "min_power_watts",
            "max_power_watts",
            "std_power_watts",
            "energy_per_query_j",
        ]
        has_energy_data = any(key in results for key in energy_keys)

        if has_energy_data:
            energy_data = metadata.copy()
            for key in energy_keys:
                if key in results:
                    energy_data[key] = results[key]

            energy_file = model_dir / "energy.json"
            with open(energy_file, "w") as f:
                json.dump(energy_data, f, indent=2)
            logger.info(f"Saved energy results: {energy_file}")

        # Save accuracy JSON files (one per task)
        for task in self.args.accuracy_tasks:
            accuracy_key = f"{task}_accuracy"
            if accuracy_key in results:
                accuracy_data = metadata.copy()
                accuracy_data["task"] = task
                accuracy_data["accuracy"] = results[accuracy_key]

                accuracy_file = model_dir / f"{task}.json"
                with open(accuracy_file, "w") as f:
                    json.dump(accuracy_data, f, indent=2)
                logger.info(f"Saved {task} accuracy results: {accuracy_file}")

        # Log to wandb
        if self.wandb_run:
            self._log_to_wandb(results, model_name, precision)

    def run_all_benchmarks(self):
        """Run benchmarks for all models with the specified precision."""
        if not self.benchmarking_available:
            logger.error(
                "Cannot run benchmarks: Heavy dependencies not available. \n"
                "Install with: pip install -e '.[full]'"
            )
            return []

        precision = self.args.precision  # Single precision now

        logger.info("#" * 70)
        logger.info("SLiM-Eval: COMPLETE BENCHMARK SUITE")
        logger.info("#" * 70)
        logger.info(f"Models: {len(self.args.models)}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Total configs: {len(self.args.models)}")
        logger.info("Metrics: Latency, Memory, Energy, Accuracy")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Estimated time: ~{len(self.args.models) * 30} minutes")

        all_results = []

        for model_name in self.args.models:
            config_id = f"{model_name.split('/')[-1]}_{precision}"
            results = self.run_complete_benchmark(model_name, precision)
            if results:
                all_results.append(results)
                logger.info(f"All benchmarks completed for {config_id}")
            clear_cache()
            time.sleep(5)

        logger.info("#" * 70)
        logger.info("All benchmarks complete!")
        logger.info("#" * 70)

        # Finish wandb run
        if self.wandb_run:
            wandb.finish()
            logger.info("Weights & Biases run finished")
        return all_results

    def analyze_results(self):
        """Analyze and visualize results."""
        self.analyzer.analyze_results()
        self.analyzer.analyze_results()
