import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm


class QuantizationExperiment:
    """Base class for quantization experiments"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = {}
    
    def quantize_model(self, model_name: str, output_dir: Path):
        """Implement quantization logic"""
        raise NotImplementedError
    
    def measure_model_size(self, model_path: Path) -> float:
        """Measure model size in GB"""
        if not model_path.exists():
            return 0.0
        
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 ** 3)  # Convert to GB


class BitsAndBytesExperiment(QuantizationExperiment):
    """Experiment with BitsAndBytes quantization"""
    
    def __init__(self):
        super().__init__(
            "BitsAndBytes", 
            "On-the-fly quantization using BitsAndBytes library"
        )
    
    def quantize_model(self, model_name: str, output_dir: Path):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"BitsAndBytes quantization: {model_name}")
        
        # Test different BnB configurations
        configs = {
            "int8": BitsAndBytesConfig(load_in_8bit=True),
            "nf4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            ),
            "fp4": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16
            )
        }
        
        results = {}
        
        for config_name, config in configs.items():
            try:
                print(f"  Testing {config_name}...")
                start_time = time.time()
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=config,
                    device_map="auto"
                )
                
                load_time = time.time() - start_time
                
                # Test generation
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                test_input = "Hello, world!"
                inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
                
                gen_start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10)
                gen_time = time.time() - gen_start
                
                # Memory usage
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
                
                results[config_name] = {
                    "load_time_s": load_time,
                    "generation_time_s": gen_time,
                    "memory_mb": memory_mb,
                    "success": True
                }
                
                print(f"    ✓ {config_name}: Load={load_time:.2f}s, Gen={gen_time:.3f}s, Mem={memory_mb:.1f}MB")
                
                del model
                del tokenizer
                
            except Exception as e:
                results[config_name] = {
                    "error": str(e),
                    "success": False
                }
                print(f"    ✗ {config_name}: {e}")
            
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.results[model_name] = results
        return results


class LLMCompressorExperiment(QuantizationExperiment):
    """Experiment with LLM Compressor (SparseML)"""
    
    def __init__(self):
        super().__init__(
            "LLMCompressor",
            "Static quantization using LLM Compressor with GPTQ"
        )
    
    def quantize_model(self, model_name: str, output_dir: Path):
        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import GPTQModifier
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import load_dataset
            
            print(f"LLM Compressor quantization: {model_name}")
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype="auto", 
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare calibration data
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
            
            def preprocess(example):
                return {"text": example["text"]}
            
            dataset = dataset.map(preprocess)
            
            def tokenize(sample):
                return tokenizer(
                    sample["text"],
                    padding=False,
                    max_length=512,
                    truncation=True
                )
            
            dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
            
            # Test different GPTQ configurations
            gptq_configs = {
                "w8a8": GPTQModifier(targets="Linear", scheme="W8A8"),
                "w4a16": GPTQModifier(targets="Linear", scheme="W4A16"),
            }
            
            results = {}
            
            for config_name, modifier in gptq_configs.items():
                try:
                    print(f"  Testing {config_name}...")
                    start_time = time.time()
                    
                    # Apply quantization
                    oneshot(
                        model=model,
                        dataset=dataset,
                        recipe=modifier,
                        max_seq_length=512,
                        num_calibration_samples=100
                    )
                    
                    quant_time = time.time() - start_time
                    
                    # Save quantized model
                    config_output_dir = output_dir / f"{model_name.split('/')[-1]}_{config_name}"
                    config_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    model.save_pretrained(config_output_dir)
                    tokenizer.save_pretrained(config_output_dir)
                    
                    # Measure size
                    model_size_gb = self.measure_model_size(config_output_dir)
                    
                    results[config_name] = {
                        "quantization_time_s": quant_time,
                        "model_size_gb": model_size_gb,
                        "output_path": str(config_output_dir),
                        "success": True
                    }
                    
                    print(f"    ✓ {config_name}: Time={quant_time:.1f}s, Size={model_size_gb:.2f}GB")
                    
                except Exception as e:
                    results[config_name] = {
                        "error": str(e),
                        "success": False
                    }
                    print(f"    ✗ {config_name}: {e}")
            
            self.results[model_name] = results
            return results
            
        except ImportError:
            print("LLM Compressor not available")
            return {"error": "LLM Compressor not installed"}
        except Exception as e:
            print(f"LLM Compressor experiment failed: {e}")
            return {"error": str(e)}


class OptimumExperiment(QuantizationExperiment):
    """Experiment with Optimum quantization"""
    
    def __init__(self):
        super().__init__(
            "Optimum",
            "Quantization using HuggingFace Optimum"
        )
    
    def quantize_model(self, model_name: str, output_dir: Path):
        try:
            from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            from transformers import AutoTokenizer
            
            print(f"Optimum quantization: {model_name}")
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Test different quantization approaches
            quant_configs = {
                "dynamic_int8": AutoQuantizationConfig.avx512_vnni(is_static=False),
                "static_int8": AutoQuantizationConfig.avx512_vnni(is_static=True),
            }
            
            results = {}
            
            for config_name, config in quant_configs.items():
                try:
                    print(f"  Testing {config_name}...")
                    start_time = time.time()
                    
                    # Convert to ONNX and quantize
                    model = ORTModelForCausalLM.from_pretrained(
                        model_name,
                        export=True
                    )
                    
                    quantizer = ORTQuantizer.from_pretrained(model)
                    quantizer.quantize(save_dir=output_dir / config_name, quantization_config=config)
                    
                    quant_time = time.time() - start_time
                    model_size_gb = self.measure_model_size(output_dir / config_name)
                    
                    results[config_name] = {
                        "quantization_time_s": quant_time,
                        "model_size_gb": model_size_gb,
                        "success": True
                    }
                    
                    print(f"    ✓ {config_name}: Time={quant_time:.1f}s, Size={model_size_gb:.2f}GB")
                    
                except Exception as e:
                    results[config_name] = {
                        "error": str(e),
                        "success": False
                    }
                    print(f"    ✗ {config_name}: {e}")
            
            self.results[model_name] = results
            return results
            
        except ImportError:
            print("Optimum not available")
            return {"error": "Optimum not installed"}
        except Exception as e:
            print(f"Optimum experiment failed: {e}")
            return {"error": str(e)}


class QuantizationComparison:
    """Compare different quantization approaches"""
    
    def __init__(self, output_dir: str = "quantization_experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiments = [
            BitsAndBytesExperiment(),
            LLMCompressorExperiment(),
            # OptimumExperiment(),  # Uncomment if Optimum is available
        ]
        
        print(f"Quantization Comparison initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Experiments: {[exp.name for exp in self.experiments]}")
    
    def run_comparison(self, model_names: List[str]):
        """Run all quantization experiments on given models"""
        all_results = {}
        
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"QUANTIZATION COMPARISON: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            for experiment in self.experiments:
                try:
                    print(f"\nRunning {experiment.name} experiment...")
                    exp_output_dir = self.output_dir / experiment.name.lower()
                    exp_output_dir.mkdir(exist_ok=True)
                    
                    results = experiment.quantize_model(model_name, exp_output_dir)
                    model_results[experiment.name] = results
                    
                except Exception as e:
                    print(f"Experiment {experiment.name} failed: {e}")
                    model_results[experiment.name] = {"error": str(e)}
                
                finally:
                    # Cleanup between experiments
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(2)
            
            all_results[model_name] = model_results
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = "quantization_comparison.json"):
        """Save comparison results"""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    def generate_summary(self, results: Dict):
        """Generate a summary of quantization experiments"""
        print(f"\n{'='*60}")
        print("QUANTIZATION EXPERIMENTS SUMMARY")
        print(f"{'='*60}")
        
        for model_name, model_results in results.items():
            print(f"\nModel: {model_name}")
            print("-" * 40)
            
            for exp_name, exp_results in model_results.items():
                print(f"\n{exp_name}:")
                
                if "error" in exp_results:
                    print(f"  ✗ Failed: {exp_results['error']}")
                    continue
                
                for config_name, config_results in exp_results.items():
                    if isinstance(config_results, dict):
                        if config_results.get("success", False):
                            size = config_results.get("model_size_gb", "N/A")
                            time_taken = config_results.get("quantization_time_s", 
                                                          config_results.get("load_time_s", "N/A"))
                            print(f"  ✓ {config_name}: Size={size}GB, Time={time_taken}s")
                        else:
                            error = config_results.get("error", "Unknown error")
                            print(f"  ✗ {config_name}: {error}")


def main():
    """Main experimental runner"""
    comparison = QuantizationComparison()
    
    # Test with smaller models for quick experiments
    test_models = [
        "microsoft/DialoGPT-small",
        # "microsoft/Phi-3-mini-4k-instruct",  # Uncomment for larger tests
    ]
    
    # Run comparison
    results = comparison.run_comparison(test_models)
    
    # Save and summarize results
    comparison.save_results(results)
    comparison.generate_summary(results)
    
    print(f"\n{'='*60}")
    print("Quantization Experiments Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()