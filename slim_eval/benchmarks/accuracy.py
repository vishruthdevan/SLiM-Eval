try:
    import wandb
except ImportError:
    wandb = None
import json
import logging
from pathlib import Path
from typing import Any, Dict

from lm_eval import evaluator as lm_evaluator

from ..utils import clear_cache
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class AccuracyBenchmark(BaseBenchmark):
    """Benchmark for measuring model accuracy on various tasks."""

    def __init__(self, args, quantized_models_dir: Path, output_dir: Path = None):
        """Initialize accuracy benchmark.

        Args:
            args: Arguments object containing benchmark settings.
            quantized_models_dir: Directory where quantized models are stored.
            output_dir: Directory where results should be saved (optional).
        """
        super().__init__(args)
        self.quantized_models_dir = quantized_models_dir
        self.output_dir = output_dir

    def get_result_keys(self) -> list:
        """Get the list of result keys this benchmark produces."""
        return [f"{task}_accuracy" for task in self.args.accuracy_tasks]

    def save_task_result(
        self,
        task_name: str,
        accuracy: float,
        model_name: str,
        precision: str,
        metadata: Dict[str, Any],
    ):
        """Save individual task result to JSON file immediately.

        Args:
            task_name: Name of the accuracy task.
            accuracy: Accuracy score for the task.
            model_name: Name or path of the model.
            precision: Precision mode being evaluated.
            metadata: Additional metadata to include in the JSON file.
        """
        if self.output_dir is None:
            return

        # Create model_precision directory
        model_short_name = model_name.split("/")[-1]
        model_dir = self.output_dir / f"{model_short_name}_{precision}"
        model_dir.mkdir(exist_ok=True)

        # Prepare task result data
        task_data = metadata.copy()
        task_data["task"] = task_name
        task_data["accuracy"] = accuracy

        # Save to JSON file
        task_file = model_dir / f"{task_name}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)
        logger.info(f"Saved {task_name} accuracy result: {task_file}")

    def run(
        self, llm, model_name: str, precision: str, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
                    task_accuracy = 0.0

                    if "exact_match,flexible-extract" in task_res:
                        task_accuracy = task_res["exact_match,flexible-extract"]
                        logger.info(
                            f"Using 'exact_match,flexible-extract': {task_accuracy}"
                        )
                        metric_found = True
                    elif "exact_match,strict-match" in task_res:
                        task_accuracy = task_res["exact_match,strict-match"]
                        logger.info(
                            f"Using 'exact_match,strict-match': {task_accuracy}"
                        )
                        metric_found = True
                    elif "exact_match" in task_res:
                        task_accuracy = task_res["exact_match"]
                        logger.info(f"Using 'exact_match': {task_accuracy}")
                        metric_found = True
                    elif "acc_norm" in task_res:
                        task_accuracy = task_res["acc_norm"]
                        logger.info(f"Using 'acc_norm': {task_accuracy}")
                        metric_found = True
                    elif "acc" in task_res:
                        task_accuracy = task_res["acc"]
                        logger.info(f"Using 'acc': {task_accuracy}")
                        metric_found = True
                    elif "pass@1" in task_res:
                        task_accuracy = task_res["pass@1"]
                        logger.info(f"Using 'pass@1': {task_accuracy}")
                        metric_found = True

                    if not metric_found:
                        for key, value in task_res.items():
                            if isinstance(value, (int, float)) and 0 <= value <= 1:
                                task_accuracy = value
                                logger.info(f"Using fallback metric '{key}': {value}")
                                metric_found = True
                                break

                    if not metric_found:
                        logger.warning(
                            f"No valid metric found for '{task_name}'. Setting accuracy to 0."
                        )
                        task_accuracy = 0.0

                    # Store result
                    accuracy_results[f"{task_name}_accuracy"] = task_accuracy

                    # Save task result immediately after completion
                    if metadata:
                        self.save_task_result(
                            task_name, task_accuracy, model_name, precision, metadata
                        )
                    
                    # Log to wandb if available
                    try:
                        if wandb.run is not None:
                            wandb.log({
                                f"accuracy/{task_name}_progress": task_accuracy,
                                f"accuracy/{task_name}_progress_pct": task_accuracy * 100,
                            })
                    except Exception:
                        pass  # wandb not initialized, skip

            logger.info("Accuracy results:")
            for task, acc in accuracy_results.items():
                logger.info(f"  {task}: {acc:.4f} ({acc * 100:.2f}%)")
            return accuracy_results
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}", exc_info=True)
            return {f"{task}_accuracy": 0 for task in self.args.accuracy_tasks}
