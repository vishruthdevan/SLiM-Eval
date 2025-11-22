"""
Quantize models using llm-compressor for SLiM-Eval
Run this before benchmarking to prepare quantized models
"""

from pathlib import Path

from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Same model list as benchmark script
MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Quantization schemes
QUANTIZATION_CONFIGS = {
    "int8": {
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True},
                "targets": ["Linear"],
            }
        }
    },
    "int4": {
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 128,
                },
                "targets": ["Linear"],
            }
        }
    },
    "gptq": {
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "group_size": 128,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head"],
    },
}

OUTPUT_BASE_DIR = Path("quantized_models")
OUTPUT_BASE_DIR.mkdir(exist_ok=True)

# Calibration dataset for GPTQ
CALIBRATION_DATASET = "wikitext"
CALIBRATION_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 512

# =============================================================================
# QUANTIZATION FUNCTIONS
# =============================================================================


def quantize_model(model_name: str, precision: str, output_dir: Path):
    """
    Quantize a model using llm-compressor.

    Args:
        model_name: HuggingFace model identifier
        precision: One of ["int8", "int4", "gptq"]
        output_dir: Directory to save quantized model
    """
    print(f"\n{'=' * 60}")
    print(f"Quantizing {model_name} to {precision.upper()}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Skip if already quantized
    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"✓ Already quantized, skipping...")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Select quantization config
        if precision not in QUANTIZATION_CONFIGS:
            print(f"✗ Unsupported precision: {precision}")
            return

        quant_config = QUANTIZATION_CONFIGS[precision]

        # For GPTQ, use GPTQModifier with calibration data
        if precision == "gptq":
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"],
            )

            # Oneshot quantization with calibration
            oneshot(
                model=model_name,
                dataset=CALIBRATION_DATASET,
                dataset_config_name=CALIBRATION_SPLIT,
                num_calibration_samples=NUM_CALIBRATION_SAMPLES,
                recipe=recipe,
                output_dir=str(output_dir),
                trust_remote_code=True,
            )

        # For INT8/INT4, use QuantizationModifier
        else:
            recipe = QuantizationModifier(**quant_config)

            # Oneshot quantization (no calibration needed for symmetric)
            oneshot(
                model=model_name,
                dataset=CALIBRATION_DATASET,
                dataset_config_name=CALIBRATION_SPLIT,
                num_calibration_samples=NUM_CALIBRATION_SAMPLES,
                recipe=recipe,
                output_dir=str(output_dir),
                trust_remote_code=True,
            )

        print(f"✓ Quantization complete: {output_dir}")

    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Quantize all models in all precisions."""
    print(f"\n{'#' * 60}")
    print("SLiM-Eval: Model Quantization")
    print(f"{'#' * 60}\n")
    print(f"Models: {len(MODELS)}")
    print(f"Precisions: {list(QUANTIZATION_CONFIGS.keys())}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")

    for model_name in MODELS:
        model_short_name = model_name.split("/")[-1]

        for precision in QUANTIZATION_CONFIGS.keys():
            output_dir = OUTPUT_BASE_DIR / f"{model_short_name}_{precision}"
            quantize_model(model_name, precision, output_dir)

    print(f"\n{'#' * 60}")
    print("Quantization Complete")
    print(f"{'#' * 60}")
    print(f"Quantized models saved to: {OUTPUT_BASE_DIR}")
    print("\nNext step: Update benchmark script to use these quantized models")


if __name__ == "__main__":
    main()
