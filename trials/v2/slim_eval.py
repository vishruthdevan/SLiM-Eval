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