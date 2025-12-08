"""Main SLiM-Eval orchestrator - refactored modular version."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from vllm import LLM

from .analysis import ResultsAnalyzer
from .benchmarks.accuracy import AccuracyBenchmark
from .benchmarks.energy import EnergyBenchmark
from .benchmarks.performance import PerformanceBenchmark
from .quantization import QuantizationManager
from .utils import clear_cache, get_model_size

logger = logging.getLogger(__name__)


class SLiMEvaluator:
    """Main evaluator orchestrating quantization and benchmarking."""

    def __init__(self, args):
        """Initialize SLiM-Eval.

        Args:
            args: Arguments object containing all configuration.
        """
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.quantized_models_dir = Path(args.quantized_models_dir)
        self.quantized_models_dir.mkdir(exist_ok=True)

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
        logger.info(f"Precisions: {args.precisions}")

        if torch.cuda.is_available():
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
        else:
            logger.warning("CUDA not available")

        # Initialize components
        self.quantization_manager = QuantizationManager(args)
        self.performance_benchmark = PerformanceBenchmark(args)
        self.energy_benchmark = EnergyBenchmark(args)
        self.accuracy_benchmark = AccuracyBenchmark(args, self.quantized_models_dir)
        self.analyzer = ResultsAnalyzer(self.output_dir, args.accuracy_tasks)

    def setup_vllm_model(
        self, model_name: str, precision: str, use_quantized_dir: bool = True
    ) -> Optional[LLM]:
        """Setup and load a vLLM model.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Precision mode (fp16, int8, int4, gptq).
            use_quantized_dir: Whether to use pre-quantized models from disk.

        Returns:
            Loaded vLLM instance or None if loading failed.
        """
        clear_cache()
        try:
            logger.info("=" * 60)
            logger.info(f"Loading {model_name} in {precision.upper()} precision...")
            logger.info("=" * 60)

            if precision == "fp16":
                model_path = model_name
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
                model_short_name = model_name.split("/")[-1]
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
                    time.sleep(2)  # Give time for memory to be fully released

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
                            precision if precision in ["int8", "int4", "gptq"] else None
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

    def run_complete_benchmark(self, model_name: str, precision: str) -> Optional[Dict]:
        """Run complete benchmark suite for a model configuration.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Precision mode to evaluate.

        Returns:
            Dictionary of results or None if benchmark failed.
        """
        logger.info("#" * 70)
        logger.info(f"COMPLETE BENCHMARK: {model_name} ({precision.upper()})")
        logger.info("#" * 70)

        model_size_info = get_model_size(model_name)
        logger.info(
            f"Model size: {model_size_info['num_parameters_b']:.2f}B parameters "
            f"({model_size_info['size_gb_fp16']:.2f} GB in FP16)"
        )

        # Get quantization scheme name
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
            "size_gb_fp16": model_size_info["size_gb_fp16"],
        }

        llm = self.setup_vllm_model(model_name, precision)
        if llm is None:
            logger.error("Model setup failed, skipping...")
            return None

        try:
            # Run performance benchmark (latency & memory)
            if self.args.enable_performance_tracking:
                results.update(
                    self.performance_benchmark.run(llm, model_name, precision)
                )

            # Run energy benchmark
            if self.args.enable_energy_tracking:
                results.update(self.energy_benchmark.run(llm, model_name, precision))

            # Clean up vLLM instance before accuracy testing
            del llm
            clear_cache()
            time.sleep(2)

            # Run accuracy benchmark (creates its own vLLM instance)
            if self.args.enable_accuracy_tracking:
                results.update(self.accuracy_benchmark.run(None, model_name, precision))
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
            "size_gb_fp16": results.get("size_gb_fp16"),
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

    def run_all_benchmarks(self):
        """Run benchmarks for all model and precision combinations."""
        logger.info("#" * 70)
        logger.info("SLiM-Eval: COMPLETE BENCHMARK SUITE")
        logger.info("#" * 70)
        logger.info(f"Models: {len(self.args.models)}")
        logger.info(f"Precisions: {self.args.precisions}")
        logger.info(
            f"Total configs: {len(self.args.models) * len(self.args.precisions)}"
        )
        logger.info("Metrics: Latency, Memory, Energy, Accuracy")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(
            f"Estimated time: ~{len(self.args.models) * len(self.args.precisions) * 30} minutes"
        )

        all_results = []

        for model_name in self.args.models:
            for precision in self.args.precisions:
                config_id = f"{model_name.split('/')[-1]}_{precision}"
                results = self.run_complete_benchmark(model_name, precision)
                if results:
                    all_results.append(results)
                    self.save_results_to_json(model_name, precision, results)
                    logger.info(f"Results saved for {config_id}")
                clear_cache()
                time.sleep(5)

        logger.info("#" * 70)
        logger.info("ALL BENCHMARKS COMPLETE!")
        logger.info("#" * 70)
        return all_results

    def analyze_results(self):
        """Analyze and visualize results."""
        self.analyzer.analyze_results()
