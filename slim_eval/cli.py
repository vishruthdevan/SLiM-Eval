import logging
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
    gptq = "gptq"


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
    models: Annotated[List[str], typer.Option(help="HF model ids or local paths")],
    precisions: Annotated[
        List[Precision],
        typer.Option(
            help="Precisions to evaluate: fp16 (baseline), int8 (SmoothQuant+GPTQ W8A8), int4 (SmoothQuant+GPTQ W4A16), gptq (W4A16 GPTQ-only)"
        ),
    ] = [Precision.fp16, Precision.int8, Precision.int4, Precision.gptq],
    output_dir: Annotated[
        str, typer.Option(help="Directory to write results")
    ] = "outputs",
    quantized_models_dir: Annotated[
        str, typer.Option(help="Directory to store/load pre-quantized models")
    ] = "quantized-models",
    num_warmup: Annotated[
        int, typer.Option(help="Warmup requests before measuring")
    ] = 2,
    num_runs: Annotated[int, typer.Option(help="Number of measured requests")] = 10,
    batch_size: Annotated[
        int, typer.Option(help="Concurrent requests per iteration")
    ] = 1,
    prompt: Annotated[
        str, typer.Option(help="Prompt used for latency tests")
    ] = "Hello, world!",
    max_new_tokens: Annotated[
        int, typer.Option(help="Max tokens to generate per request")
    ] = 128,
    gpu_memory_utilization: Annotated[
        float, typer.Option(help="vLLM GPU memory utilization fraction")
    ] = 0.9,
    max_model_len: Annotated[
        int, typer.Option(help="vLLM max model length (context window)")
    ] = 2048,
    tasks: Annotated[
        List[Task],
        typer.Option(
            help="Benchmark tasks to run. Options: performance (latency & memory), energy (power consumption), accuracy (model quality)"
        ),
    ] = [Task.performance],
    energy_sample_runs: Annotated[
        int, typer.Option(help="Number of energy-tracked requests")
    ] = 10,
    accuracy_tasks: Annotated[List[str], typer.Option(help="lm-eval tasks to run")] = [
        "mmlu",
        "gsm8k",
        "hellaswag",
    ],
    num_fewshot: Annotated[
        int, typer.Option(help="Few-shot examples for accuracy tasks")
    ] = 0,
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
    ] = "train",
    num_calibration_samples: Annotated[
        int, typer.Option(help="Number of calibration samples")
    ] = 512,
    max_sequence_length: Annotated[
        int, typer.Option(help="Max sequence length for calibration")
    ] = 2048,
):
    """Run complete benchmarks across models and precisions."""
    # Convert task list to boolean flags
    tasks_list = [t.value for t in tasks]
    enable_performance_tracking = "performance" in tasks_list
    enable_energy_tracking = "energy" in tasks_list
    enable_accuracy_tracking = "accuracy" in tasks_list

    args = _build_args(
        models=list(models),
        precisions=[p.value for p in precisions],
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


@app.command()
def analyze(
    output_dir: Annotated[
        str, typer.Option(help="Directory containing results")
    ] = "outputs",
    accuracy_tasks: Annotated[
        List[str], typer.Option(help="Accuracy tasks to include in analysis")
    ] = ["mmlu", "gsm8k", "hellaswag"],
):
    """Analyze and visualize previously saved results."""
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
    app()
