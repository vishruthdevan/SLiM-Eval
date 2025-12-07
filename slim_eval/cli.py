import logging
from pathlib import Path
from typing import List

import click

from .evaluator import SLiMEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group(
    help="SLiM-Eval: Quantize and benchmark LLMs for speed, memory, energy, and accuracy."
)
def cli():
    pass


def _build_args(**kwargs):
    class Args:
        pass

    args = Args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


@cli.command("run", help="Run complete benchmarks across models and precisions")
@click.option(
    "--models", multiple=True, required=True, help="HF model ids or local paths"
)
@click.option(
    "--precisions",
    multiple=True,
    type=click.Choice(["fp16", "int8", "int4", "gptq"], case_sensitive=False),
    default=["fp16", "int8", "int4", "gptq"],
    show_default=True,
    help="Precisions to evaluate: fp16 (baseline), int8 (SmoothQuant+GPTQ W8A8), int4 (SmoothQuant+GPTQ W4A16), gptq (W4A16 GPTQ-only).",
)
@click.option(
    "--output-dir",
    default="outputs",
    show_default=True,
    help="Directory to write results",
)
@click.option(
    "--quantized-models-dir",
    default="quantized-models",
    show_default=True,
    help="Directory to store/load pre-quantized models",
)
@click.option(
    "--num-warmup",
    default=2,
    show_default=True,
    help="Warmup requests before measuring",
)
@click.option(
    "--num-runs", default=10, show_default=True, help="Number of measured requests"
)
@click.option(
    "--batch-size",
    default=1,
    show_default=True,
    help="Concurrent requests per iteration",
)
@click.option(
    "--prompt",
    default="Hello, world!",
    show_default=True,
    help="Prompt used for latency tests",
)
@click.option(
    "--max-new-tokens",
    default=128,
    show_default=True,
    help="Max tokens to generate per request",
)
@click.option(
    "--gpu-memory-utilization",
    default=0.9,
    show_default=True,
    help="vLLM GPU memory utilization fraction",
)
@click.option(
    "--max-model-len",
    default=8192,
    show_default=True,
    help="vLLM max model length (context window)",
)
@click.option(
    "--tasks",
    multiple=True,
    type=click.Choice(["performance", "energy", "accuracy"], case_sensitive=False),
    default=["performance"],
    show_default=True,
    help="Benchmark tasks to run. Options: performance (latency & memory), energy (power consumption), accuracy (model quality)",
)
@click.option(
    "--energy-sample-runs",
    default=10,
    show_default=True,
    help="Number of energy-tracked requests",
)
@click.option(
    "--accuracy-tasks",
    multiple=True,
    default=["mmlu", "gsm8k", "hellaswag"],
    show_default=True,
    help="lm-eval tasks to run",
)
@click.option(
    "--num-fewshot",
    default=0,
    show_default=True,
    help="Few-shot examples for accuracy tasks",
)
@click.option(
    "--accuracy-limit",
    default=None,
    type=int,
    help="Limit examples per task for quick runs",
)
@click.option(
    "--accuracy-batch-size",
    default=32,
    show_default=True,
    help="Global lm-eval batch size",
)
@click.option(
    "--accuracy-batch-size-hellaswag",
    default=32,
    type=int,
    help="Override batch size for hellaswag",
)
@click.option(
    "--accuracy-batch-size-gsm8k",
    default=32,
    type=int,
    help="Override batch size for gsm8k",
)
@click.option(
    "--accuracy-batch-size-mmlu",
    default=32,
    type=int,
    help="Override batch size for mmlu",
)
@click.option(
    "--calibration-dataset", default="HuggingFaceH4/ultrachat_200k", show_default=True
)
@click.option("--calibration-split", default="train", show_default=True)
@click.option("--num-calibration-samples", default=512, show_default=True)
@click.option("--max-sequence-length", default=2048, show_default=True)
def run(
    models: List[str],
    precisions: List[str],
    output_dir: str,
    quantized_models_dir: str,
    num_warmup: int,
    num_runs: int,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    tasks: List[str],
    energy_sample_runs: int,
    accuracy_tasks: List[str],
    num_fewshot: int,
    accuracy_limit: int,
    accuracy_batch_size: int,
    accuracy_batch_size_hellaswag: int,
    accuracy_batch_size_gsm8k: int,
    accuracy_batch_size_mmlu: int,
    calibration_dataset: str,
    calibration_split: str,
    num_calibration_samples: int,
    max_sequence_length: int,
):
    # Convert task list to boolean flags
    tasks_list = [t.lower() for t in tasks]
    enable_performance_tracking = "performance" in tasks_list
    enable_energy_tracking = "energy" in tasks_list
    enable_accuracy_tracking = "accuracy" in tasks_list

    args = _build_args(
        models=list(models),
        precisions=list(precisions),
        output_dir=output_dir,
        quantized_models_dir=quantized_models_dir,
        num_warmup=num_warmup,
        num_runs=num_runs,
        batch_size=batch_size,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_performance_tracking=enable_performance_tracking,
        enable_energy_tracking=enable_energy_tracking,
        energy_sample_runs=energy_sample_runs,
        enable_accuracy_tracking=enable_accuracy_tracking,
        accuracy_tasks=list(accuracy_tasks),
        num_fewshot=num_fewshot,
        accuracy_limit=accuracy_limit,
        accuracy_batch_size=accuracy_batch_size,
        accuracy_batch_size_hellaswag=accuracy_batch_size_hellaswag,
        accuracy_batch_size_gsm8k=accuracy_batch_size_gsm8k,
        accuracy_batch_size_mmlu=accuracy_batch_size_mmlu,
        calibration_dataset=calibration_dataset,
        calibration_split=calibration_split,
        num_calibration_samples=num_calibration_samples,
        max_sequence_length=max_sequence_length,
    )

    evaluator = SLiMEvaluator(args)
    evaluator.run_all_benchmarks()


@cli.command("analyze", help="Analyze and visualize previously saved results")
@click.option("--output-dir", default="outputs", show_default=True)
@click.option(
    "--accuracy-tasks",
    multiple=True,
    default=["mmlu", "gsm8k", "hellaswag"],
    show_default=True,
    help="Accuracy tasks to include in analysis",
)
def analyze(output_dir: str, accuracy_tasks: List[str]):
    args = _build_args(
        models=[],
        precisions=[],
        output_dir=output_dir,
        quantized_models_dir="quantized-models",
        num_warmup=0,
        num_runs=0,
        batch_size=1,
        prompt="",
        max_new_tokens=0,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_performance_tracking=False,
        enable_energy_tracking=False,
        energy_sample_runs=0,
        enable_accuracy_tracking=False,
        accuracy_tasks=list(accuracy_tasks),
        num_fewshot=0,
        accuracy_limit=None,
        accuracy_batch_size=0,
        accuracy_batch_size_hellaswag=None,
        accuracy_batch_size_gsm8k=None,
        accuracy_batch_size_mmlu=None,
        calibration_dataset="",
        calibration_split="",
        num_calibration_samples=0,
        max_sequence_length=0,
    )
    evaluator = SLiMEvaluator(args)
    evaluator.analyze_results()


def main():
    cli()
