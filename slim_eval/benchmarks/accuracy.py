"""Accuracy benchmarking for SLiM-Eval."""

import logging
from pathlib import Path
from typing import Any, Dict

from lm_eval import evaluator as lm_evaluator

from ..utils import clear_cache
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class AccuracyBenchmark(BaseBenchmark):
    """Benchmark for measuring model accuracy on various tasks."""

    def __init__(self, args, quantized_models_dir: Path):
        """Initialize accuracy benchmark.

        Args:
            args: Arguments object containing benchmark settings.
            quantized_models_dir: Directory where quantized models are stored.
        """
        super().__init__(args)
        self.quantized_models_dir = quantized_models_dir

    def get_result_keys(self) -> list:
        """Get the list of result keys this benchmark produces."""
        return [f"{task}_accuracy" for task in self.args.accuracy_tasks]

    def run(self, llm, model_name: str, precision: str) -> Dict[str, Any]:
        """Run accuracy benchmark.

        Note: This benchmark doesn't use the llm parameter as it creates its own
        vLLM instance through lm-eval.

        Args:
            llm: Unused (lm-eval creates its own model instance).
            model_name: Name or path of the model.
            precision: Precision mode being evaluated.

        Returns:
            Dictionary containing accuracy metrics for each task.
        """
        logger.info("=" * 60)
        logger.info("ACCURACY EVALUATION")
        logger.info("=" * 60)
        logger.info(f"Tasks: {', '.join(self.args.accuracy_tasks)}")
        logger.info(f"Few-shot: {self.args.num_fewshot}")
        if self.args.accuracy_limit:
            logger.info(f"Limit: {self.args.accuracy_limit} examples per task")

        # Aggressive memory clearing before accuracy evaluation
        logger.info("Clearing GPU memory before accuracy evaluation...")
        clear_cache()

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

            def resolve_batch_size(task: str) -> int:
                tl = task.lower()
                if (
                    tl == "hellaswag"
                    and self.args.accuracy_batch_size_hellaswag is not None
                ):
                    return self.args.accuracy_batch_size_hellaswag
                if tl == "gsm8k" and self.args.accuracy_batch_size_gsm8k is not None:
                    return self.args.accuracy_batch_size_gsm8k
                if tl == "mmlu" and self.args.accuracy_batch_size_mmlu is not None:
                    return self.args.accuracy_batch_size_mmlu
                return self.args.accuracy_batch_size

            accuracy_results = {}
            for task in self.args.accuracy_tasks:
                bs = resolve_batch_size(task)
                logger.info(f"Running accuracy task '{task}' with batch_size={bs}")

                results = lm_evaluator.simple_evaluate(
                    model="vllm",
                    model_args=model_args,
                    tasks=[task],
                    num_fewshot=self.args.num_fewshot,
                    batch_size=bs,
                    limit=self.args.accuracy_limit,
                    log_samples=False,
                )

                for task_name, task_res in results["results"].items():
                    logger.info(f"Task '{task_name}' metrics: {list(task_res.keys())}")
                    logger.info(f"Task '{task_name}' values: {task_res}")

                    metric_found = False

                    if "exact_match,flexible-extract" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res[
                            "exact_match,flexible-extract"
                        ]
                        logger.info(
                            f"Using 'exact_match,flexible-extract': {task_res['exact_match,flexible-extract']}"
                        )
                        metric_found = True
                    elif "exact_match,strict-match" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res[
                            "exact_match,strict-match"
                        ]
                        logger.info(
                            f"Using 'exact_match,strict-match': {task_res['exact_match,strict-match']}"
                        )
                        metric_found = True
                    elif "exact_match" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res[
                            "exact_match"
                        ]
                        logger.info(f"Using 'exact_match': {task_res['exact_match']}")
                        metric_found = True
                    elif "acc_norm" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res["acc_norm"]
                        logger.info(f"Using 'acc_norm': {task_res['acc_norm']}")
                        metric_found = True
                    elif "acc" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res["acc"]
                        logger.info(f"Using 'acc': {task_res['acc']}")
                        metric_found = True
                    elif "pass@1" in task_res:
                        accuracy_results[f"{task_name}_accuracy"] = task_res["pass@1"]
                        logger.info(f"Using 'pass@1': {task_res['pass@1']}")
                        metric_found = True

                    if not metric_found:
                        for key, value in task_res.items():
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                accuracy_results[f"{task_name}_accuracy"] = value
                                logger.info(f"Using fallback metric '{key}': {value}")
                                metric_found = True
                                break

                    if not metric_found:
                        logger.warning(
                            f"No valid metric found for '{task_name}'. Setting accuracy to 0."
                        )
                        accuracy_results[f"{task_name}_accuracy"] = 0.0

            logger.info("Accuracy results:")
            for task, acc in accuracy_results.items():
                logger.info(f"  {task}: {acc:.4f} ({acc * 100:.2f}%)")
            return accuracy_results
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}", exc_info=True)
            return {f"{task}_accuracy": 0 for task in self.args.accuracy_tasks}
