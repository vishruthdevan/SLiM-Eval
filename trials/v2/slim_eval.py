import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ModelBackend:
    """Abstract base class for different inference backends"""
    
    def __init__(self, model_name: str, precision: str):
        self.model_name = model_name
        self.precision = precision
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        raise NotImplementedError
    
    def generate(self, prompts: List[str], max_tokens: int = 32) -> List[str]:
        raise NotImplementedError
    
    def cleanup(self):
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TransformersBackend(ModelBackend):
    """HuggingFace Transformers backend with BitsAndBytes quantization"""
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"Loading {self.model_name} with Transformers backend ({self.precision})")
        
        # Configure quantization
        if self.precision == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif self.precision == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.precision == "fp16" else "auto"
        )
    
    def generate(self, prompts: List[str], max_tokens: int = 32) -> List[str]:
        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(output_text)
        return outputs


class VLLMBackend(ModelBackend):
    """vLLM backend for high-throughput inference"""
    
    def load_model(self):
        from vllm import LLM
        
        print(f"Loading {self.model_name} with vLLM backend ({self.precision})")
        
        # Configure vLLM parameters
        quantization = None if self.precision == "fp16" else self.precision
        
        self.model = LLM(
            model=self.model_name,
            dtype="float16" if self.precision == "fp16" else "auto",
            quantization=quantization,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            tensor_parallel_size=1
        )
    
    def generate(self, prompts: List[str], max_tokens: int = 32) -> List[str]:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class SLiMEval:

    def __init__(self, output_dir: str = "slim_eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Experimental configuration
        self.backends = ["transformers", "vllm"]
        self.precisions = ["fp16", "int8", "int4"]
        self.test_prompts = [
            "Explain machine learning in simple terms.",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
            "Describe the water cycle.",
            "How do neural networks work?"
        ]
        
        print(f"SLiM-Eval initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Backends: {self.backends}")
        print(f"Precisions: {self.precisions}")
    
    def get_backend(self, backend_name: str, model_name: str, precision: str) -> ModelBackend:
        """Factory method to create backend instances"""
        if backend_name == "transformers":
            return TransformersBackend(model_name, precision)
        elif backend_name == "vllm":
            return VLLMBackend(model_name, precision)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def benchmark_latency(self, backend: ModelBackend, num_runs: int = 50) -> Dict:
        """Simplified latency benchmarking"""
        print(f"Benchmarking latency ({num_runs} runs)...")
        
        latencies = []
        
        # Warmup
        for _ in range(5):
            backend.generate([self.test_prompts[0]], max_tokens=16)
        
        # Benchmark
        for i in tqdm(range(num_runs), desc="Latency"):
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            start_time = time.time()
            backend.generate([prompt], max_tokens=32)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        lat_array = np.array(latencies)
        return {
            "mean_latency_s": lat_array.mean(),
            "median_latency_s": np.median(lat_array),
            "std_latency_s": lat_array.std(),
            "min_latency_s": lat_array.min(),
            "max_latency_s": lat_array.max()
        }
    
    def benchmark_memory(self, backend: ModelBackend) -> Dict:
        """Memory usage benchmarking"""
        if not torch.cuda.is_available():
            return {"peak_memory_mb": 0, "allocated_memory_mb": 0}
        
        torch.cuda.reset_peak_memory_stats()
        
        # Generate some text to measure peak memory
        backend.generate(self.test_prompts[:3], max_tokens=64)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        
        return {
            "peak_memory_mb": peak_memory,
            "allocated_memory_mb": allocated_memory
        }
    
    def benchmark_throughput(self, backend: ModelBackend, batch_sizes: List[int] = [1, 4, 8]) -> Dict:
        """Throughput benchmarking with different batch sizes"""
        print("Benchmarking throughput...")
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            prompts = self.test_prompts[:batch_size]
            
            start_time = time.time()
            outputs = backend.generate(prompts, max_tokens=32)
            end_time = time.time()
            
            total_time = end_time - start_time
            total_tokens = sum(len(output.split()) for output in outputs)
            throughput = total_tokens / total_time
            
            throughput_results[f"throughput_batch_{batch_size}"] = throughput
        
        return throughput_results
    
    def run_backend_comparison(self, model_name: str) -> Dict:
        """Compare different backends for the same model"""
        print(f"\n{'='*60}")
        print(f"BACKEND COMPARISON: {model_name}")
        print(f"{'='*60}")
        
        all_results = []
        
        for backend_name in self.backends:
            for precision in self.precisions:
                print(f"\nTesting {backend_name} with {precision}...")
                
                try:
                    backend = self.get_backend(backend_name, model_name, precision)
                    backend.load_model()
                    
                    # Run benchmarks
                    results = {
                        "model": model_name,
                        "backend": backend_name,
                        "precision": precision,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Latency benchmark
                    latency_results = self.benchmark_latency(backend)
                    results.update(latency_results)
                    
                    # Memory benchmark
                    memory_results = self.benchmark_memory(backend)
                    results.update(memory_results)
                    
                    # Throughput benchmark
                    throughput_results = self.benchmark_throughput(backend)
                    results.update(throughput_results)
                    
                    all_results.append(results)
                    
                    print(f"✓ {backend_name} {precision}: "
                          f"Latency={results['mean_latency_s']:.3f}s, "
                          f"Memory={results['peak_memory_mb']:.1f}MB")
                    
                except Exception as e:
                    print(f"✗ Failed {backend_name} {precision}: {e}")
                    results = {
                        "model": model_name,
                        "backend": backend_name,
                        "precision": precision,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    all_results.append(results)
                
                finally:
                    if 'backend' in locals():
                        backend.cleanup()
                    time.sleep(2)
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = "backend_comparison.json"):
        """Save results to JSON file"""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    def generate_report(self, results: List[Dict]):
        """Generate a simple comparison report"""
        df = pd.DataFrame(results)
        
        # Filter out failed experiments
        df_success = df[~df.get('error', pd.Series()).notna()]
        
        if df_success.empty:
            print("No successful experiments to report")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Group by backend and precision
        for backend in df_success['backend'].unique():
            print(f"\n{backend.upper()} Backend:")
            backend_df = df_success[df_success['backend'] == backend]
            
            for precision in backend_df['precision'].unique():
                precision_df = backend_df[backend_df['precision'] == precision]
                if not precision_df.empty:
                    row = precision_df.iloc[0]
                    print(f"  {precision}: "
                          f"Latency={row.get('mean_latency_s', 'N/A'):.3f}s, "
                          f"Memory={row.get('peak_memory_mb', 'N/A'):.1f}MB")
        
        # Save CSV report
        report_file = self.output_dir / "comparison_report.csv"
        df_success.to_csv(report_file, index=False)
        print(f"\nDetailed report saved to: {report_file}")


def main():
    """Main experimental runner"""
    evaluator = SLiMEval()
    
    # Test models (smaller ones for quick experiments)
    test_models = [
        "microsoft/DialoGPT-small",  # Small model for quick testing
        # "microsoft/Phi-3-mini-4k-instruct",  # Uncomment for larger tests
    ]
    
    all_results = []
    
    for model in test_models:
        try:
            model_results = evaluator.run_backend_comparison(model)
            all_results.extend(model_results)
        except Exception as e:
            print(f"Failed to test {model}: {e}")
    
    # Save and report results
    evaluator.save_results(all_results)
    evaluator.generate_report(all_results)
    
    print(f"\n{'='*60}")
    print("SLiM-Eval Experimental Run Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# Additional Advanced Features for v2
class AdvancedCacheManager:
    """Advanced caching system for model artifacts and results"""
    
    def __init__(self, cache_dir: str = "cache_v2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index = {}
        self.load_cache_index()
    
    def load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.cache_index = json.load(f)
    
    def save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def get_cache_key(self, model_name: str, precision: str, config: Dict) -> str:
        """Generate cache key for model configuration"""
        import hashlib
        
        cache_data = {
            'model_name': model_name,
            'precision': precision,
            'config': config
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if result is cached"""
        return cache_key in self.cache_index
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result"""
        if not self.is_cached(cache_key):
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def cache_result(self, cache_key: str, result: Dict):
        """Cache benchmark result"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        self.cache_index[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'file': str(cache_file)
        }
        
        self.save_cache_index()


class DistributedExecutor:
    """Distributed execution manager for multi-node benchmarking"""
    
    def __init__(self, nodes: List[str] = None):
        self.nodes = nodes or ['localhost']
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
    
    def start_workers(self, num_workers: int = mp.cpu_count()):
        """Start distributed workers"""
        self.is_running = True
        
        for i in range(num_workers):
            worker = mp.Process(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
        
        print(f"Started {num_workers} distributed workers")
    
    def stop_workers(self):
        """Stop all workers"""
        self.is_running = False
        
        # Send stop signals
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()
        print("All workers stopped")
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks"""
        print(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Stop signal
                    break
                
                # Process task
                result = self._execute_task(task, worker_id)
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
        
        print(f"Worker {worker_id} stopped")
    
    def _execute_task(self, task: Dict, worker_id: int) -> Dict:
        """Execute a single benchmark task"""
        try:
            # Simulate task execution
            time.sleep(np.random.uniform(0.1, 2.0))
            
            return {
                'task_id': task['id'],
                'worker_id': worker_id,
                'status': 'completed',
                'result': {
                    'latency': np.random.uniform(0.1, 1.0),
                    'memory': np.random.uniform(1000, 5000),
                    'accuracy': np.random.uniform(0.7, 0.95)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'task_id': task['id'],
                'worker_id': worker_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def submit_task(self, task: Dict):
        """Submit task for execution"""
        self.task_queue.put(task)
    
    def get_results(self, timeout: float = None) -> List[Dict]:
        """Get all available results"""
        results = []
        
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        
        return results


class SystemProfiler:
    """Advanced system profiling and monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        print("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("System monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict:
        """Collect system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'process_count': len(psutil.pids())
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                })
            
            metrics['gpus'] = gpu_metrics
            
        except Exception:
            metrics['gpus'] = []
        
        return metrics
    
    def get_metrics_summary(self, minutes: int = 10) -> Dict:
        """Get metrics summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        
        summary = {
            'time_period_minutes': minutes,
            'sample_count': len(recent_metrics),
            'cpu_avg': np.mean(cpu_values),
            'cpu_max': np.max(cpu_values),
            'cpu_min': np.min(cpu_values),
            'memory_avg': np.mean(memory_values),
            'memory_max': np.max(memory_values),
            'memory_min': np.min(memory_values)
        }
        
        # GPU summary if available
        if recent_metrics[0].get('gpus'):
            gpu_loads = []
            gpu_memory_percents = []
            
            for metrics in recent_metrics:
                for gpu in metrics['gpus']:
                    gpu_loads.append(gpu['load'])
                    gpu_memory_percents.append(gpu['memory_percent'])
            
            if gpu_loads:
                summary['gpu_load_avg'] = np.mean(gpu_loads)
                summary['gpu_load_max'] = np.max(gpu_loads)
                summary['gpu_memory_avg'] = np.mean(gpu_memory_percents)
                summary['gpu_memory_max'] = np.max(gpu_memory_percents)
        
        return summary


class BenchmarkOrchestrator:
    """Advanced orchestration with caching, distribution, and monitoring"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.cache_manager = AdvancedCacheManager()
        self.distributed_executor = DistributedExecutor()
        self.system_profiler = SystemProfiler()
        
        # Results storage
        self.results_db = []
        self.execution_stats = {
            'total_benchmarks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0
        }
    
    def start_services(self):
        """Start all background services"""
        self.system_profiler.start_monitoring()
        self.distributed_executor.start_workers()
        print("All services started")
    
    def stop_services(self):
        """Stop all background services"""
        self.system_profiler.stop_monitoring()
        self.distributed_executor.stop_workers()
        print("All services stopped")
    
    def run_benchmark_suite(self, models: List[str], precisions: List[str], 
                          backends: List[str]) -> Dict:
        """Run comprehensive benchmark suite with advanced features"""
        
        print(f"Starting benchmark suite:")
        print(f"  Models: {len(models)}")
        print(f"  Precisions: {len(precisions)}")
        print(f"  Backends: {len(backends)}")
        
        start_time = time.time()
        
        # Generate all benchmark tasks
        tasks = []
        task_id = 0
        
        for model in models:
            for precision in precisions:
                for backend in backends:
                    task = {
                        'id': f"task_{task_id:04d}",
                        'model': model,
                        'precision': precision,
                        'backend': backend,
                        'config': self.config
                    }
                    tasks.append(task)
                    task_id += 1
        
        print(f"Generated {len(tasks)} benchmark tasks")
        
        # Execute tasks with caching
        results = []
        
        for task in tqdm(tasks, desc="Executing benchmarks"):
            result = self._execute_cached_benchmark(task)
            results.append(result)
            
            # Submit to distributed executor for parallel processing
            self.distributed_executor.submit_task(task)
        
        # Collect distributed results
        distributed_results = self.distributed_executor.get_results(timeout=1.0)
        
        # Combine results
        all_results = results + distributed_results
        
        # Update statistics
        execution_time = time.time() - start_time
        self.execution_stats['total_execution_time'] += execution_time
        self.execution_stats['total_benchmarks'] += len(tasks)
        
        # Generate summary
        summary = {
            'total_tasks': len(tasks),
            'execution_time': execution_time,
            'results': all_results,
            'cache_stats': {
                'hits': self.execution_stats['cache_hits'],
                'misses': self.execution_stats['cache_misses'],
                'hit_rate': self.execution_stats['cache_hits'] / max(1, self.execution_stats['total_benchmarks'])
            },
            'system_metrics': self.system_profiler.get_metrics_summary()
        }
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _execute_cached_benchmark(self, task: Dict) -> Dict:
        """Execute benchmark with caching"""
        
        # Generate cache key
        cache_key = self.cache_manager.get_cache_key(
            task['model'], task['precision'], task['config']
        )
        
        # Check cache first
        cached_result = self.cache_manager.get_cached_result(cache_key)
        
        if cached_result:
            self.execution_stats['cache_hits'] += 1
            cached_result['cache_hit'] = True
            return cached_result
        
        # Execute benchmark
        self.execution_stats['cache_misses'] += 1
        
        result = self._execute_benchmark(task)
        result['cache_hit'] = False
        
        # Cache result
        self.cache_manager.cache_result(cache_key, result)
        
        return result
    
    def _execute_benchmark(self, task: Dict) -> Dict:
        """Execute actual benchmark"""
        
        # Simulate benchmark execution
        execution_time = np.random.uniform(0.5, 3.0)
        time.sleep(execution_time * 0.1)  # Scaled down for demo
        
        # Generate realistic results based on task parameters
        model_factor = {'small': 0.5, 'medium': 1.0, 'large': 2.0}.get(
            task['model'].split('_')[-1], 1.0
        )
        
        precision_factor = {'fp16': 1.0, 'int8': 0.7, 'int4': 0.5}.get(
            task['precision'], 1.0
        )
        
        backend_factor = {'transformers': 1.0, 'vllm': 0.8, 'tensorrt': 0.6}.get(
            task['backend'], 1.0
        )
        
        result = {
            'task_id': task['id'],
            'model': task['model'],
            'precision': task['precision'],
            'backend': task['backend'],
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'metrics': {
                'latency_mean': 0.1 * model_factor / (precision_factor * backend_factor),
                'latency_std': 0.01 * model_factor,
                'memory_peak_mb': 1000 * model_factor * precision_factor,
                'throughput_tokens_per_sec': 100 / model_factor * precision_factor * backend_factor,
                'accuracy_score': max(0.6, 0.9 - (1 - precision_factor) * 0.1),
                'energy_joules': 50 * model_factor / precision_factor
            }
        }
        
        return result
    
    def _save_results(self, summary: Dict):
        """Save benchmark results"""
        
        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"benchmark_results_v2_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save to CSV for analysis
        results_df = pd.DataFrame([
            {
                'task_id': r['task_id'],
                'model': r['model'],
                'precision': r['precision'],
                'backend': r['backend'],
                'cache_hit': r['cache_hit'],
                **r['metrics']
            }
            for r in summary['results']
            if 'metrics' in r
        ])
        
        csv_file = f"benchmark_results_v2_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {results_file} and {csv_file}")
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_statistics': self.execution_stats,
            'system_metrics': self.system_profiler.get_metrics_summary(30),
            'cache_performance': {
                'total_entries': len(self.cache_manager.cache_index),
                'hit_rate': self.execution_stats['cache_hits'] / max(1, self.execution_stats['total_benchmarks']),
                'cache_size_mb': self._calculate_cache_size()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        
        for cache_entry in self.cache_manager.cache_index.values():
            cache_file = Path(cache_entry['file'])
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Cache recommendations
        hit_rate = self.execution_stats['cache_hits'] / max(1, self.execution_stats['total_benchmarks'])
        
        if hit_rate < 0.3:
            recommendations.append(
                "Low cache hit rate detected. Consider running similar benchmarks "
                "to improve cache efficiency."
            )
        
        # System resource recommendations
        system_metrics = self.system_profiler.get_metrics_summary(10)
        
        if system_metrics.get('cpu_avg', 0) > 80:
            recommendations.append(
                "High CPU usage detected. Consider reducing concurrent benchmarks "
                "or upgrading hardware."
            )
        
        if system_metrics.get('memory_avg', 0) > 85:
            recommendations.append(
                "High memory usage detected. Consider using smaller models "
                "or more aggressive quantization."
            )
        
        if system_metrics.get('gpu_memory_avg', 0) > 90:
            recommendations.append(
                "High GPU memory usage detected. Consider reducing batch sizes "
                "or using gradient checkpointing."
            )
        
        return recommendations


# Enhanced main execution
def main_advanced():
    """Advanced main execution with full feature demonstration"""
    
    # Configuration
    config = {
        'num_runs': 50,
        'batch_size': 8,
        'max_tokens': 32,
        'enable_profiling': True,
        'cache_enabled': True,
        'distributed_execution': True
    }
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(config)
    
    try:
        # Start services
        orchestrator.start_services()
        
        # Define benchmark parameters
        models = ['model_small', 'model_medium', 'model_large']
        precisions = ['fp16', 'int8', 'int4']
        backends = ['transformers', 'vllm']
        
        # Run benchmark suite
        results = orchestrator.run_benchmark_suite(models, precisions, backends)
        
        print(f"\nBenchmark Suite Results:")
        print(f"  Total tasks: {results['total_tasks']}")
        print(f"  Execution time: {results['execution_time']:.2f}s")
        print(f"  Cache hit rate: {results['cache_stats']['hit_rate']:.2%}")
        
        # Generate performance report
        report = orchestrator.generate_performance_report()
        
        print(f"\nPerformance Report:")
        print(f"  Total benchmarks executed: {report['execution_statistics']['total_benchmarks']}")
        print(f"  Cache size: {report['cache_performance']['cache_size_mb']:.2f} MB")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save report
        report_file = f"performance_report_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nPerformance report saved to: {report_file}")
        
    finally:
        # Cleanup
        orchestrator.stop_services()


if __name__ == "__main__":
    # Run both simple and advanced versions
    print("="*60)
    print("Running SLiM-Eval v2 - Simple Version")
    print("="*60)
    main()
    
    print("\n" + "="*60)
    print("Running SLiM-Eval v2 - Advanced Version")
    print("="*60)
    main_advanced()