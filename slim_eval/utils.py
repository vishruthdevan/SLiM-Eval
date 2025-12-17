"""Shared utility functions for SLiM-Eval."""

import gc
import logging
from typing import Dict

import torch
import torch.distributed as dist

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


def clear_cache() -> None:
    """Clear GPU cache and garbage collect."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            logger.debug(f"Error destroying process group: {e}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_mb(gpu_index: int = 0) -> float:
    """Get current GPU memory usage in MB.

    Returns:
        Current GPU memory usage in megabytes.
    """
    if torch.cuda.is_available():
        try:
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.used / (1024**2)
            else:
                return torch.cuda.memory_allocated(gpu_index) / (1024**2)
        except Exception:
            return torch.cuda.memory_allocated(gpu_index) / (1024**2)
    return 0.0


def get_peak_gpu_memory_mb(gpu_index: int = 0) -> float:
    """Get peak GPU memory usage in MB.

    Returns:
        Peak GPU memory usage in megabytes.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(gpu_index) / (1024**2)
    return 0.0


def get_model_size(model_path: str) -> Dict[str, float]:
    """Get model size information.

    Args:
        model_path: Path to the model.

    Returns:
        Dictionary with num_parameters, num_parameters_b, and size_gb_fp16.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_params = 0
        if hasattr(config, "num_parameters"):
            num_params = config.num_parameters
        elif hasattr(config, "n_params"):
            num_params = config.n_params
        else:
            if hasattr(config, "hidden_size") and hasattr(config, "num_hidden_layers"):
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, "vocab_size", 32000)
                embed_params = vocab_size * hidden_size
                layer_params = num_layers * (12 * hidden_size * hidden_size)
                head_params = vocab_size * hidden_size
                num_params = embed_params + layer_params + head_params

        num_params_b = num_params / 1e9
        size_gb_fp16 = (num_params * 2) / (1024**3)
        return {
            "num_parameters": num_params,
            "num_parameters_b": num_params_b,
            "size_gb_fp16": size_gb_fp16,
        }
    except Exception as e:
        logger.warning(f"Failed to get model size for {model_path}: {e}")
        return {"num_parameters": 0, "num_parameters_b": 0, "size_gb_fp16": 0}


def get_quantized_model_size(model_dir: str) -> float:
    """Get the actual size of a quantized model from disk.

    Calculates the total size of model weight files (safetensors, bin, pt files)
    in the model directory.

    Args:
        model_dir: Path to the quantized model directory.

    Returns:
        Size of the model in GB.
    """
    from pathlib import Path

    model_path = Path(model_dir)
    if not model_path.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return 0.0

    # Model weight file extensions
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth"}

    total_size_bytes = 0
    for file in model_path.iterdir():
        if file.is_file() and file.suffix in weight_extensions:
            total_size_bytes += file.stat().st_size

    size_gb = total_size_bytes / (1024**3)
    logger.debug(f"Quantized model size for {model_dir}: {size_gb:.4f} GB")
    return size_gb
