"""Performance benchmarking (latency and memory) for SLiM-Eval."""

import logging
import time
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..utils import clear_cache, get_gpu_memory_mb, get_peak_gpu_memory_mb
from .base import BaseBenchmark

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceBenchmark(BaseBenchmark):
    """Benchmark for measuring inference performance (latency and memory usage)."""

    def get_result_keys(self) -> list:
        """Get the list of result keys this benchmark produces."""
        return [
            "mean_latency_s",
            "median_latency_s",
            "p95_latency_s",
            "p99_latency_s",
            "std_latency_s",
            "mean_peak_mem_mb",
            "mean_avg_mem_mb",
            "tokens_per_second",
            "baseline_memory_mb",
        ]

    def run(self, llm: LLM, model_name: str, precision: str) -> Dict[str, Any]:
        """Run latency and memory benchmark.

        Args:
            llm: Loaded vLLM model instance.
            model_name: Name or path of the model.
            precision: Precision mode being evaluated.

        Returns:
            Dictionary containing latency and memory metrics.
        """
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.args.max_new_tokens, top_p=1.0
        )
        latencies, peak_memories, avg_memories, tokens_generated = [], [], [], []
        total_iterations = (
            self.args.num_runs + self.args.num_warmup + self.args.batch_size - 1
        ) // self.args.batch_size

        logger.info("=" * 60)
        logger.info("PERFORMANCE BENCHMARK (LATENCY & MEMORY)")
        logger.info("=" * 60)
        logger.info(
            f"Warmup: {self.args.num_warmup} | Benchmark: {self.args.num_runs} | Batch: {self.args.batch_size}"
        )

        nvml_available = False
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                nvml_available = True
            except Exception as e:
                logger.warning(f"Could not initialize NVML: {e}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            baseline_memory_mb = get_gpu_memory_mb(self.args.gpu_index)
            logger.info(
                f"Baseline GPU memory (model + KV cache): {baseline_memory_mb:.2f} MB"
            )
        else:
            baseline_memory_mb = 0

        iteration_count = 0
        pbar = tqdm(
            total=self.args.num_runs + self.args.num_warmup, desc="Latency/Memory"
        )
        for _ in range(total_iterations):
            current_batch_size = min(
                self.args.batch_size,
                self.args.num_runs + self.args.num_warmup - iteration_count,
            )
            prompts = [self.args.prompt] * current_batch_size

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            batch_latency = end_time - start_time
            peak_mem = get_peak_gpu_memory_mb(self.args.gpu_index)
            avg_mem = get_gpu_memory_mb(self.args.gpu_index)
            per_request_latency = batch_latency / current_batch_size
            batch_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

            for _ in range(current_batch_size):
                if iteration_count >= self.args.num_warmup:
                    latencies.append(per_request_latency)
                    peak_memories.append(
                        peak_mem if peak_mem > 0 else baseline_memory_mb
                    )
                    avg_memories.append(avg_mem if avg_mem > 0 else baseline_memory_mb)
                    tokens_generated.append(batch_tokens / current_batch_size)

                iteration_count += 1
                pbar.update(1)
                if iteration_count >= self.args.num_runs + self.args.num_warmup:
                    break

            if iteration_count >= self.args.num_runs + self.args.num_warmup:
                break
        pbar.close()

        if nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        lat = np.array(latencies)
        pm = np.array(peak_memories)
        am = np.array(avg_memories)
        tg = np.array(tokens_generated)

        results = {
            "mean_latency_s": lat.mean(),
            "median_latency_s": np.median(lat),
            "p95_latency_s": np.percentile(lat, 95),
            "p99_latency_s": np.percentile(lat, 99),
            "std_latency_s": lat.std(),
            "mean_peak_mem_mb": pm.mean(),
            "mean_avg_mem_mb": am.mean(),
            "tokens_per_second": tg.mean() / lat.mean(),
            "baseline_memory_mb": baseline_memory_mb,
        }

        logger.info(
            f"Latency: {results['mean_latency_s']:.4f}s | Memory: {results['mean_peak_mem_mb']:.2f}MB"
        )

        # Clear GPU memory after performance benchmark
        clear_cache()

        return results
