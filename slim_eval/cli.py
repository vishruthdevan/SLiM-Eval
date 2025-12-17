import logging
import os
from enum import Enum
from typing import List, Optional

import typer
from typing_extensions import Annotated

from .evaluator import SLiMEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="SLiM-Eval: Quantize and benchmark LLMs for speed, memory, energy, and accuracy."
)


class Precision(str, Enum):
    """Available precision modes."""

    fp16 = "fp16"
    int8 = "int8"
    int4 = "int4"


class Task(str, Enum):
    """Available benchmark tasks."""

    performance = "performance"
    energy = "energy"
    accuracy = "accuracy"


def _build_args(**kwargs):
    class Args:
        pass

    args = Args()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


@app.command()
def run(
    models: Annotated[
        str,
        typer.Option(help="Space-separated HF model ids or local paths"),
    ] = "meta-llama/Llama-3.2-3B-Instruct",
    precision: Annotated[
        Precision,
        typer.Option(
            help="Precision mode: fp16 (baseline), int8 (GPTQ W8A8), int4 (GPTQ W4A16)"
        ),
    ] = Precision.fp16,
    output_dir: Annotated[
        str, typer.Option(help="Directory to write results")
    ] = "outputs",
    quantized_models_dir: Annotated[
        str, typer.Option(help="Directory to write quantized models to")
    ] = "quantized-models",
    num_warmup: Annotated[int, typer.Option(help="Warmup iterations")] = 10,
    num_runs: Annotated[int, typer.Option(help="Measurement iterations")] = 500,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for performance tests")
    ] = 8,
    prompt: Annotated[
        str, typer.Option(help="Prompt to use for performance and energy tests")
    ] = "Explain one interesting fact about large language models.",
    max_new_tokens: Annotated[
        int, typer.Option(help="Max new tokens to generate")
    ] = 256,
    gpu_memory_utilization: Annotated[
        float, typer.Option(help="GPU memory utilization fraction")
    ] = 0.8,
    max_model_len: Annotated[
        int, typer.Option(help="Maximum model context length")
    ] = 8192,
    tasks: Annotated[
        str,
        typer.Option(
            help="Space-separated tasks to run: performance energy accuracy (or any combination)"
        ),
    ] = "performance accuracy energy",
    energy_sample_runs: Annotated[
        int, typer.Option(help="Number of energy sample runs")
    ] = 200,
    accuracy_tasks: Annotated[
        str,
        typer.Option(
            help="Space-separated accuracy tasks: mmlu gsm8k hellaswag (or any combination)"
        ),
    ] = "mmlu gsm8k hellaswag",
    num_fewshot: Annotated[
        int, typer.Option(help="Number of few-shot examples for accuracy tests")
    ] = 5,
    accuracy_limit: Annotated[
        Optional[int], typer.Option(help="Limit examples per task for quick runs")
    ] = None,
    accuracy_batch_size: Annotated[
        int, typer.Option(help="Global lm-eval batch size")
    ] = 32,
    accuracy_batch_size_hellaswag: Annotated[
        int, typer.Option(help="Override batch size for hellaswag")
    ] = 32,
    accuracy_batch_size_gsm8k: Annotated[
        int, typer.Option(help="Override batch size for gsm8k")
    ] = 32,
    accuracy_batch_size_mmlu: Annotated[
        int, typer.Option(help="Override batch size for mmlu")
    ] = 32,
    calibration_dataset: Annotated[
        str, typer.Option(help="Calibration dataset for quantization")
    ] = "HuggingFaceH4/ultrachat_200k",
    calibration_split: Annotated[
        str, typer.Option(help="Dataset split for calibration")
    ] = "train_sft",
    num_calibration_samples: Annotated[
        int, typer.Option(help="Number of calibration samples")
    ] = 512,
    max_sequence_length: Annotated[
        int, typer.Option(help="Max sequence length for calibration")
    ] = 2048,
    gpu_index: Annotated[
        int, typer.Option(help="Select NVIDIA GPU index to use (0-based)")
    ] = 0,
    wandb_enabled: Annotated[
        bool, typer.Option(help="Enable Weights & Biases logging")
    ] = True,
    wandb_project: Annotated[str, typer.Option(help="W&B project name")] = "slim-eval",
    wandb_api_key: Annotated[
        str, typer.Option(help="W&B API key (or set WANDB_API_KEY env var)")
    ] = "",
    wandb_run_name: Annotated[
        str,
        typer.Option(help="W&B run name template (leave empty for auto-generation)"),
    ] = "",
):
    """Run complete benchmarks for models with specified precision."""
    # Convert space-separated strings to lists
    models_list = models.split()
    tasks_list = tasks.split()
    accuracy_tasks_list = accuracy_tasks.split()

    # Convert task list to boolean flags
    enable_performance_tracking = "performance" in tasks_list
    enable_energy_tracking = "energy" in tasks_list
    enable_accuracy_tracking = "accuracy" in tasks_list

    args = _build_args(
        models=models_list,
        precision=precision.value,
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
        accuracy_tasks=accuracy_tasks_list,
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
        gpu_index=gpu_index,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_api_key=wandb_api_key,
        wandb_run_name=wandb_run_name,
    )

    evaluator = SLiMEvaluator(args)
    evaluator.run_all_benchmarks()


@app.command()
def analyze(
    input_dir: Annotated[
        str, typer.Option(help="Directory containing benchmark results to analyze")
    ] = "outputs",
    output_dir: Annotated[
        str,
        typer.Option(help="Directory to write analysis results (plots, CSVs, etc.)"),
    ] = "outputs",
    accuracy_tasks: Annotated[
        List[str], typer.Option(help="Accuracy tasks to include in analysis")
    ] = ["mmlu", "gsm8k", "hellaswag"],
    gpu_index: Annotated[
        int, typer.Option(help="Select NVIDIA GPU index to use (0-based)")
    ] = 0,
):
    """Analyze and visualize previously saved results."""
    args = _build_args(
        models=[],
        precision="fp16",
        input_dir=input_dir,
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
        gpu_index=gpu_index,
    )
    evaluator = SLiMEvaluator(args)
    evaluator.analyze_results()


def main():
    app()
