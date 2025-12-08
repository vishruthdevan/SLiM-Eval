"""Quantization logic for SLiM-Eval."""

import gc
import logging
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure torch dynamo to support quantization with scalar outputs
torch._dynamo.config.capture_scalar_outputs = True

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Manages model quantization using llm-compressor."""

    def __init__(self, args):
        """Initialize quantization manager.

        Args:
            args: Arguments object containing quantization settings.
        """
        self.args = args
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

    def get_quantization_config(self, precision: str) -> Dict:
        """Get quantization configuration for a precision mode.

        Args:
            precision: Precision mode (int8, int4, gptq).

        Returns:
            Quantization configuration dictionary.
        """
        return self.quantization_configs.get(precision, {})

    def quantize_model(self, model_name: str, precision: str, output_dir: Path) -> None:
        """Quantize a model and save it.

        Args:
            model_name: HuggingFace model name or local path.
            precision: Target precision (int8, int4, gptq).
            output_dir: Directory to save the quantized model.
        """
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

            logger.info("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

            logger.info(f"Loading calibration dataset: {self.args.calibration_dataset}")
            ds = load_dataset(
                self.args.calibration_dataset,
                split=f"{self.args.calibration_split}[:{self.args.num_calibration_samples}]",
            )
            ds = ds.shuffle(seed=42)

            def preprocess(example):
                try:
                    return {
                        "text": tokenizer.apply_chat_template(
                            example["messages"], tokenize=False
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

            def tokenize(sample):
                return tokenizer(
                    sample["text"],
                    padding=False,
                    max_length=self.args.max_sequence_length,
                    truncation=True,
                    add_special_tokens=False,
                )

            ds = ds.map(tokenize, remove_columns=ds.column_names)

            quant_config = self.quantization_configs[precision]
            recipe = quant_config["recipe"]
            logger.info(f"Applying quantization recipe: {quant_config['method']}")

            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=self.args.max_sequence_length,
                num_calibration_samples=self.args.num_calibration_samples,
            )

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

            logger.info(f"Saving to {output_dir}...")
            model.save_pretrained(output_dir, save_compressed=True)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Quantization complete: {output_dir}")

            # Thorough cleanup to free GPU memory
            logger.info("Cleaning up GPU memory...")
            del model
            del tokenizer
            del ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU memory cleanup complete")
        except Exception as e:
            logger.error(f"Quantization failed: {e}", exc_info=True)
