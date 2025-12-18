import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Standardized benchmark result structure"""
    model_name: str
    precision: str
    benchmark_type: str
    timestamp: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BenchmarkComponent(ABC):
    """Abstract base class for benchmark components"""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def run_benchmark(self, model_name: str, precision: str, **kwargs) -> BenchmarkResult:
        """Run the benchmark and return results"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    def get_requirements(self) -> List[str]:
        """Return list of required packages"""
        return []


class LatencyBenchmark(BenchmarkComponent):
    """Modular latency benchmarking component"""
    
    def __init__(self, config: Dict = None):
        super().__init__("LatencyBenchmark", config)
        self.model = None
        self.tokenizer = None
    
    async def initialize(self) -> bool:
        """Initialize the latency benchmark"""
        try:
            # Check if required packages are available
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.is_initialized = True
            return True
        except ImportError as e:
            print(f"LatencyBenchmark initialization failed: {e}")
            return False
    
    async def run_benchmark(self, model_name: str, precision: str, **kwargs) -> BenchmarkResult:
        """Run latency benchmark"""
        if not self.is_initialized:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="latency",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error="Component not initialized"
            )
        
        try:
            # Load model based on precision
            await self._load_model(model_name, precision)
            
            # Configuration
            num_runs = self.config.get("num_runs", 50)
            warmup_runs = self.config.get("warmup_runs", 5)
            max_tokens = self.config.get("max_tokens", 32)
            test_prompt = self.config.get("test_prompt", "Explain machine learning briefly.")
            
            # Run benchmark
            latencies = await self._measure_latencies(test_prompt, num_runs, warmup_runs, max_tokens)
            
            # Calculate metrics
            metrics = {
                "mean_latency_s": np.mean(latencies),
                "median_latency_s": np.median(latencies),
                "std_latency_s": np.std(latencies),
                "min_latency_s": np.min(latencies),
                "max_latency_s": np.max(latencies),
                "p95_latency_s": np.percentile(latencies, 95),
                "p99_latency_s": np.percentile(latencies, 99),
                "num_runs": len(latencies)
            }
            
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="latency",
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                metadata={"config": self.config}
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="latency",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error=str(e)
            )
        finally:
            await self._cleanup_model()
    
    async def _load_model(self, model_name: str, precision: str):
        """Load model with specified precision"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        # Configure quantization
        quant_config = None
        if precision == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16 if precision == "fp16" else "auto"
        )
    
    async def _measure_latencies(self, prompt: str, num_runs: int, warmup_runs: int, max_tokens: int) -> List[float]:
        """Measure inference latencies"""
        import torch
        
        latencies = []
        
        # Warmup
        for _ in range(warmup_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Actual measurements
        for _ in tqdm(range(num_runs), desc="Measuring latency"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=max_tokens)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        return latencies
    
    async def _cleanup_model(self):
        """Cleanup model resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def cleanup(self):
        """Cleanup component"""
        await self._cleanup_model()
        self.is_initialized = False


class MemoryBenchmark(BenchmarkComponent):
    """Memory usage benchmarking component"""
    
    def __init__(self, config: Dict = None):
        super().__init__("MemoryBenchmark", config)
    
    async def initialize(self) -> bool:
        try:
            import torch
            import psutil
            self.is_initialized = True
            return True
        except ImportError:
            return False
    
    async def run_benchmark(self, model_name: str, precision: str, **kwargs) -> BenchmarkResult:
        """Run memory benchmark"""
        if not self.is_initialized:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="memory",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error="Component not initialized"
            )
        
        try:
            import torch
            import psutil
            
            # Measure baseline memory
            baseline_gpu_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            baseline_ram_mb = psutil.virtual_memory().used / (1024**2)
            
            # Load model (reuse latency benchmark logic)
            latency_benchmark = LatencyBenchmark(self.config)
            await latency_benchmark.initialize()
            await latency_benchmark._load_model(model_name, precision)
            
            # Measure peak memory during inference
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            # Run a few inferences to measure peak memory
            test_prompt = self.config.get("test_prompt", "Explain machine learning briefly.")
            for _ in range(3):
                inputs = latency_benchmark.tokenizer(test_prompt, return_tensors="pt").to(latency_benchmark.model.device)
                with torch.no_grad():
                    latency_benchmark.model.generate(**inputs, max_new_tokens=32)
            
            # Measure final memory usage
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            current_gpu_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            current_ram_mb = psutil.virtual_memory().used / (1024**2)
            
            metrics = {
                "baseline_gpu_mb": baseline_gpu_mb,
                "peak_gpu_mb": peak_gpu_mb,
                "current_gpu_mb": current_gpu_mb,
                "gpu_memory_increase_mb": current_gpu_mb - baseline_gpu_mb,
                "baseline_ram_mb": baseline_ram_mb,
                "current_ram_mb": current_ram_mb,
                "ram_memory_increase_mb": current_ram_mb - baseline_ram_mb
            }
            
            await latency_benchmark.cleanup()
            
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="memory",
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                metadata={"config": self.config}
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="memory",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error=str(e)
            )
    
    async def cleanup(self):
        self.is_initialized = False


class ThroughputBenchmark(BenchmarkComponent):
    """Throughput benchmarking component"""
    
    def __init__(self, config: Dict = None):
        super().__init__("ThroughputBenchmark", config)
    
    async def initialize(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.is_initialized = True
            return True
        except ImportError:
            return False
    
    async def run_benchmark(self, model_name: str, precision: str, **kwargs) -> BenchmarkResult:
        """Run throughput benchmark"""
        if not self.is_initialized:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="throughput",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error="Component not initialized"
            )
        
        try:
            # Use latency benchmark to load model
            latency_benchmark = LatencyBenchmark(self.config)
            await latency_benchmark.initialize()
            await latency_benchmark._load_model(model_name, precision)
            
            # Test different batch sizes
            batch_sizes = self.config.get("batch_sizes", [1, 2, 4, 8])
            test_prompts = [
                "Explain AI in one sentence.",
                "What is machine learning?",
                "Define neural networks.",
                "How does deep learning work?",
                "What are transformers in AI?",
                "Explain natural language processing.",
                "What is computer vision?",
                "Define reinforcement learning."
            ]
            
            throughput_results = {}
            
            for batch_size in batch_sizes:
                prompts = test_prompts[:batch_size]
                
                # Measure throughput
                start_time = time.time()
                total_tokens = 0
                
                for prompt in prompts:
                    inputs = latency_benchmark.tokenizer(prompt, return_tensors="pt").to(latency_benchmark.model.device)
                    with torch.no_grad():
                        outputs = latency_benchmark.model.generate(**inputs, max_new_tokens=32)
                    total_tokens += len(outputs[0])
                
                end_time = time.time()
                duration = end_time - start_time
                throughput = total_tokens / duration
                
                throughput_results[f"batch_{batch_size}_tokens_per_sec"] = throughput
                throughput_results[f"batch_{batch_size}_duration_s"] = duration
                throughput_results[f"batch_{batch_size}_total_tokens"] = total_tokens
            
            await latency_benchmark.cleanup()
            
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="throughput",
                timestamp=datetime.now().isoformat(),
                metrics=throughput_results,
                metadata={"config": self.config, "batch_sizes": batch_sizes}
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=model_name,
                precision=precision,
                benchmark_type="throughput",
                timestamp=datetime.now().isoformat(),
                metrics={},
                error=str(e)
            )
    
    async def cleanup(self):
        self.is_initialized = False


class BenchmarkPipeline:
    """Modular benchmark pipeline with streaming results"""
    
    def __init__(self, output_dir: str = "slim_eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.components = []
        self.results_stream = []
        
        print(f"SLiM-Eval  Pipeline initialized")
        print(f"Output directory: {self.output_dir}")
    
    def add_component(self, component: BenchmarkComponent):
        """Add a benchmark component to the pipeline"""
        self.components.append(component)
        print(f"Added component: {component.name}")
    
    async def initialize_pipeline(self) -> bool:
        """Initialize all components"""
        print("Initializing pipeline components...")
        
        initialized_components = []
        for component in self.components:
            if await component.initialize():
                initialized_components.append(component)
                print(f"  ✓ {component.name} initialized")
            else:
                print(f"  ✗ {component.name} failed to initialize")
        
        self.components = initialized_components
        return len(self.components) > 0
    
    async def run_pipeline(self, model_names: List[str], precisions: List[str]) -> Iterator[BenchmarkResult]:
        """Run the complete pipeline and yield results as they complete"""
        print(f"\nRunning pipeline for {len(model_names)} models, {len(precisions)} precisions")
        print(f"Components: {[c.name for c in self.components]}")
        
        total_tasks = len(model_names) * len(precisions) * len(self.components)
        completed_tasks = 0
        
        with tqdm(total=total_tasks, desc="Pipeline Progress") as pbar:
            for model_name in model_names:
                for precision in precisions:
                    print(f"\n{'='*50}")
                    print(f"Processing: {model_name} ({precision})")
                    print(f"{'='*50}")
                    
                    # Run components in parallel for this model/precision
                    tasks = []
                    for component in self.components:
                        task = component.run_benchmark(model_name, precision)
                        tasks.append(task)
                    
                    # Wait for all components to complete for this model/precision
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"Component failed: {result}")
                        else:
                            yield result
                            self.results_stream.append(result)
                            await self._save_result_immediately(result)
                        
                        completed_tasks += 1
                        pbar.update(1)
                    
                    # Cleanup between models
                    await self._cleanup_between_models()
    
    async def _save_result_immediately(self, result: BenchmarkResult):
        """Save individual result immediately (streaming approach)"""
        # Create model-specific directory
        model_dir = self.output_dir / result.model_name.replace("/", "_")
        model_dir.mkdir(exist_ok=True)
        
        # Save individual result
        filename = f"{result.benchmark_type}_{result.precision}_{int(time.time())}.json"
        result_file = model_dir / filename
        
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    async def _cleanup_between_models(self):
        """Cleanup resources between models"""
        import gc
        import torch
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Small delay to ensure cleanup
        await asyncio.sleep(1)
    
    async def cleanup_pipeline(self):
        """Cleanup all components"""
        print("\nCleaning up pipeline...")
        for component in self.components:
            await component.cleanup()
            print(f"  ✓ {component.name} cleaned up")
    
    def save_aggregated_results(self, filename: str = "aggregated_results.json"):
        """Save all results in aggregated format"""
        output_file = self.output_dir / filename
        
        aggregated = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(self.results_stream),
                "components": [c.name for c in self.components]
            },
            "results": [result.to_dict() for result in self.results_stream]
        }
        
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"Aggregated results saved to: {output_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of all results"""
        print(f"\n{'='*60}")
        print("SLIM-EVAL PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        if not self.results_stream:
            print("No results to summarize")
            return
        
        # Group results by model and precision
        summary = {}
        for result in self.results_stream:
            key = f"{result.model_name}_{result.precision}"
            if key not in summary:
                summary[key] = {}
            summary[key][result.benchmark_type] = result
        
        # Print summary table
        print(f"\n{'Model/Precision':<30} {'Latency (s)':<12} {'Memory (MB)':<12} {'Throughput':<12} {'Status'}")
        print("-" * 80)
        
        for key, benchmarks in summary.items():
            latency = benchmarks.get("latency")
            memory = benchmarks.get("memory")
            throughput = benchmarks.get("throughput")
            
            latency_val = f"{latency.metrics.get('mean_latency_s', 0):.3f}" if latency and not latency.error else "Failed"
            memory_val = f"{memory.metrics.get('peak_gpu_mb', 0):.1f}" if memory and not memory.error else "Failed"
            throughput_val = f"{throughput.metrics.get('batch_1_tokens_per_sec', 0):.1f}" if throughput and not throughput.error else "Failed"
            
            # Overall status
            failed_count = sum(1 for b in benchmarks.values() if b.error)
            status = f"{len(benchmarks)-failed_count}/{len(benchmarks)} OK"
            
            print(f"{key:<30} {latency_val:<12} {memory_val:<12} {throughput_val:<12} {status}")


async def main():
    """Main pipeline runner"""
    # Create pipeline
    pipeline = BenchmarkPipeline()
    
    # Add components with configurations
    latency_config = {"num_runs": 20, "warmup_runs": 3, "max_tokens": 32}
    memory_config = {"test_prompt": "Explain machine learning briefly."}
    throughput_config = {"batch_sizes": [1, 2, 4]}
    
    pipeline.add_component(LatencyBenchmark(latency_config))
    pipeline.add_component(MemoryBenchmark(memory_config))
    pipeline.add_component(ThroughputBenchmark(throughput_config))
    
    # Initialize pipeline
    if not await pipeline.initialize_pipeline():
        print("Failed to initialize pipeline")
        return
    
    # Test models (use smaller models for quick testing)
    test_models = [
        "microsoft/DialoGPT-small",
        # "microsoft/Phi-3-mini-4k-instruct",  # Uncomment for larger tests
    ]
    test_precisions = ["fp16", "int8"]
    
    try:
        # Run pipeline and process results as they stream in
        print(f"\nStarting pipeline execution...")
        async for result in pipeline.run_pipeline(test_models, test_precisions):
            if result.error:
                print(f"  ✗ {result.benchmark_type} failed for {result.model_name} ({result.precision}): {result.error}")
            else:
                print(f"  ✓ {result.benchmark_type} completed for {result.model_name} ({result.precision})")
        
        # Save aggregated results and generate report
        pipeline.save_aggregated_results()
        pipeline.generate_summary_report()
        
    finally:
        # Cleanup
        await pipeline.cleanup_pipeline()
    
    print(f"\n{'='*60}")
    print("SLiM-Eval Pipeline Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())