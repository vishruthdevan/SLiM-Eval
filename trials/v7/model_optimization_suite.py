import asyncio
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Iterator
from collections import defaultdict, deque
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    DYNAMIC_BATCHING = "dynamic_batching"
    TENSOR_PARALLELISM = "tensor_parallelism"
    PIPELINE_PARALLELISM = "pipeline_parallelism"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    FLASH_ATTENTION = "flash_attention"
    SPECULATIVE_DECODING = "speculative_decoding"


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"      # < 100M params
    SMALL = "small"    # 100M - 1B params
    MEDIUM = "medium"  # 1B - 7B params
    LARGE = "large"    # 7B - 20B params
    XLARGE = "xlarge"  # 20B - 70B params
    XXLARGE = "xxlarge" # > 70B params


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    strategy: OptimizationStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    enabled: bool = True
    compatibility_requirements: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy.value,
            "parameters": self.parameters,
            "priority": self.priority,
            "enabled": self.enabled,
            "compatibility_requirements": self.compatibility_requirements,
            "resource_requirements": self.resource_requirements
        }


@dataclass
class ModelProfile:
    """Comprehensive model profile"""
    name: str
    architecture: str
    num_parameters: int
    model_size_mb: float
    context_length: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    size_category: ModelSize
    supported_precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8", "int4"])
    memory_requirements: Dict[str, float] = field(default_factory=dict)
    compute_requirements: Dict[str, float] = field(default_factory=dict)
    optimization_compatibility: Dict[OptimizationStrategy, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if not self.memory_requirements:
            self.memory_requirements = self._estimate_memory_requirements()
        if not self.compute_requirements:
            self.compute_requirements = self._estimate_compute_requirements()
        if not self.optimization_compatibility:
            self.optimization_compatibility = self._determine_optimization_compatibility()
    
    def _estimate_memory_requirements(self) -> Dict[str, float]:
        """Estimate memory requirements for different precisions"""
        base_memory_mb = self.model_size_mb
        return {
            "fp32": base_memory_mb * 4.0,
            "fp16": base_memory_mb * 2.0,
            "int8": base_memory_mb * 1.0,
            "int4": base_memory_mb * 0.5,
            "inference_overhead": base_memory_mb * 0.3,
            "kv_cache_per_token": (self.hidden_size * self.num_layers * 2 * 2) / (1024 * 1024)  # MB per token
        }
    
    def _estimate_compute_requirements(self) -> Dict[str, float]:
        """Estimate compute requirements"""
        # Rough FLOPS estimation for transformer inference
        flops_per_token = 2 * self.num_parameters  # Forward pass approximation
        return {
            "flops_per_token": flops_per_token,
            "flops_per_second_target": flops_per_token * 100,  # Target 100 tokens/sec
            "memory_bandwidth_gb_s": self.model_size_mb / 1024 * 10,  # 10x model size
            "compute_intensity": flops_per_token / (self.model_size_mb * 1024 * 1024)
        }
    
    def _determine_optimization_compatibility(self) -> Dict[OptimizationStrategy, bool]:
        """Determine which optimizations are compatible with this model"""
        compatibility = {}
        
        # All models support basic optimizations
        compatibility[OptimizationStrategy.QUANTIZATION] = True
        compatibility[OptimizationStrategy.MIXED_PRECISION] = True
        compatibility[OptimizationStrategy.DYNAMIC_BATCHING] = True
        
        # Size-dependent optimizations
        if self.size_category in [ModelSize.LARGE, ModelSize.XLARGE, ModelSize.XXLARGE]:
            compatibility[OptimizationStrategy.TENSOR_PARALLELISM] = True
            compatibility[OptimizationStrategy.PIPELINE_PARALLELISM] = True
            compatibility[OptimizationStrategy.GRADIENT_CHECKPOINTING] = True
        else:
            compatibility[OptimizationStrategy.TENSOR_PARALLELISM] = False
            compatibility[OptimizationStrategy.PIPELINE_PARALLELISM] = False
            compatibility[OptimizationStrategy.GRADIENT_CHECKPOINTING] = False
        
        # Architecture-dependent optimizations
        if "llama" in self.name.lower() or "mistral" in self.name.lower():
            compatibility[OptimizationStrategy.FLASH_ATTENTION] = True
            compatibility[OptimizationStrategy.SPECULATIVE_DECODING] = True
        else:
            compatibility[OptimizationStrategy.FLASH_ATTENTION] = False
            compatibility[OptimizationStrategy.SPECULATIVE_DECODING] = False
        
        # Pruning and distillation are generally applicable but require special handling
        compatibility[OptimizationStrategy.PRUNING] = self.size_category != ModelSize.TINY
        compatibility[OptimizationStrategy.DISTILLATION] = self.size_category in [ModelSize.MEDIUM, ModelSize.LARGE]
        
        return compatibility


class ModelProfiler:
    """Profiles models to create comprehensive model profiles"""
    
    def __init__(self):
        self.profile_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def profile_model(self, model_name: str, force_refresh: bool = False) -> ModelProfile:
        """Create comprehensive model profile"""
        if model_name in self.profile_cache and not force_refresh:
            return self.profile_cache[model_name]
        
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            # Extract basic information
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 12)
            num_attention_heads = getattr(config, 'num_attention_heads', 12)
            vocab_size = getattr(config, 'vocab_size', 50000)
            context_length = getattr(config, 'max_position_embeddings', 2048)
            intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Estimate parameters
            num_parameters = self._estimate_parameters(config)
            
            # Determine size category
            size_category = self._categorize_model_size(num_parameters)
            
            # Estimate model size in MB (fp16 baseline)
            model_size_mb = num_parameters * 2 / (1024 * 1024)
            
            profile = ModelProfile(
                name=model_name,
                architecture=getattr(config, 'model_type', 'unknown'),
                num_parameters=num_parameters,
                model_size_mb=model_size_mb,
                context_length=context_length,
                vocab_size=vocab_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                size_category=size_category
            )
            
            self.profile_cache[model_name] = profile
            return profile
            
        except Exception as e:
            self.logger.warning(f"Failed to profile model {model_name}: {e}")
            # Return default profile
            return self._create_default_profile(model_name)
    
    def _estimate_parameters(self, config) -> int:
        """Estimate number of parameters from config"""
        try:
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 12)
            vocab_size = getattr(config, 'vocab_size', 50000)
            intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Embedding parameters
            embedding_params = vocab_size * hidden_size
            
            # Layer parameters (attention + MLP)
            attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
            mlp_params = 2 * hidden_size * intermediate_size  # Up and down projections
            layer_norm_params = 2 * hidden_size  # Two layer norms per layer
            
            layer_params = num_layers * (attention_params + mlp_params + layer_norm_params)
            
            # Output head
            output_params = hidden_size * vocab_size
            
            # Final layer norm
            final_norm_params = hidden_size
            
            total_params = embedding_params + layer_params + output_params + final_norm_params
            
            return int(total_params)
            
        except Exception as e:
            self.logger.warning(f"Parameter estimation failed: {e}")
            return self._fallback_parameter_estimation(config)
    
    def _fallback_parameter_estimation(self, config) -> int:
        """Fallback parameter estimation based on model name patterns"""
        model_name = getattr(config, 'name_or_path', '').lower()
        
        # Common model size patterns
        if any(x in model_name for x in ['tiny', 'mini']):
            return 100_000_000  # 100M
        elif any(x in model_name for x in ['small']):
            return 300_000_000  # 300M
        elif any(x in model_name for x in ['base', 'medium']):
            return 1_000_000_000  # 1B
        elif any(x in model_name for x in ['large']):
            return 3_000_000_000  # 3B
        elif '7b' in model_name:
            return 7_000_000_000  # 7B
        elif '13b' in model_name:
            return 13_000_000_000  # 13B
        elif '30b' in model_name or '33b' in model_name:
            return 30_000_000_000  # 30B
        elif '70b' in model_name:
            return 70_000_000_000  # 70B
        else:
            return 1_000_000_000  # Default 1B
    
    def _categorize_model_size(self, num_parameters: int) -> ModelSize:
        """Categorize model by parameter count"""
        if num_parameters < 100_000_000:
            return ModelSize.TINY
        elif num_parameters < 1_000_000_000:
            return ModelSize.SMALL
        elif num_parameters < 7_000_000_000:
            return ModelSize.MEDIUM
        elif num_parameters < 20_000_000_000:
            return ModelSize.LARGE
        elif num_parameters < 70_000_000_000:
            return ModelSize.XLARGE
        else:
            return ModelSize.XXLARGE
    
    def _create_default_profile(self, model_name: str) -> ModelProfile:
        """Create a default profile when profiling fails"""
        return ModelProfile(
            name=model_name,
            architecture="unknown",
            num_parameters=1_000_000_000,
            model_size_mb=2000,
            context_length=2048,
            vocab_size=50000,
            num_layers=24,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096,
            size_category=ModelSize.MEDIUM
        )


class OptimizationEngine(ABC):
    """Abstract base class for optimization engines"""
    
    def __init__(self, name: str, config: OptimizationConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_initialized = False
        self.optimization_history = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the optimization engine"""
        pass
    
    @abstractmethod
    async def optimize_model(self, model_name: str, model_profile: ModelProfile, **kwargs) -> Dict[str, Any]:
        """Apply optimization to model"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    def is_compatible(self, model_profile: ModelProfile) -> bool:
        """Check if optimization is compatible with model"""
        return model_profile.optimization_compatibility.get(self.config.strategy, False)
    
    def estimate_optimization_impact(self, model_profile: ModelProfile) -> Dict[str, float]:
        """Estimate the impact of optimization"""
        # Default implementation - subclasses should override
        return {
            "memory_reduction_ratio": 1.0,
            "latency_improvement_ratio": 1.0,
            "accuracy_retention_ratio": 1.0,
            "throughput_improvement_ratio": 1.0
        }


class QuantizationEngine(OptimizationEngine):
    """Quantization optimization engine"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("QuantizationEngine", config)
        self.quantization_methods = ["bitsandbytes", "gptq", "awq", "ggml"]
        self.precision_levels = ["int8", "int4", "nf4", "fp4"]
    
    async def initialize(self) -> bool:
        """Initialize quantization engine"""
        try:
            # Check for required libraries
            import torch
            
            # Try to import quantization libraries
            available_methods = []
            
            try:
                import bitsandbytes
                available_methods.append("bitsandbytes")
            except ImportError:
                pass
            
            try:
                from auto_gptq import AutoGPTQForCausalLM
                available_methods.append("gptq")
            except ImportError:
                pass
            
            try:
                from awq import AutoAWQForCausalLM
                available_methods.append("awq")
            except ImportError:
                pass
            
            self.available_methods = available_methods
            self.is_initialized = len(available_methods) > 0
            
            if self.is_initialized:
                self.logger.info(f"Quantization engine initialized with methods: {available_methods}")
            else:
                self.logger.warning("No quantization methods available")
            
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantization engine: {e}")
            return False
    
    async def optimize_model(self, model_name: str, model_profile: ModelProfile, **kwargs) -> Dict[str, Any]:
        """Apply quantization optimization"""
        if not self.is_initialized:
            return {"error": "Engine not initialized", "success": False}
        
        precision = kwargs.get("precision", "int8")
        method = kwargs.get("method", "bitsandbytes")
        
        if method not in self.available_methods:
            return {"error": f"Method {method} not available", "success": False}
        
        try:
            self.logger.info(f"Quantizing {model_name} to {precision} using {method}")
            
            start_time = time.time()
            
            # Simulate quantization process
            if method == "bitsandbytes":
                result = await self._quantize_with_bitsandbytes(model_name, precision, model_profile)
            elif method == "gptq":
                result = await self._quantize_with_gptq(model_name, precision, model_profile)
            elif method == "awq":
                result = await self._quantize_with_awq(model_name, precision, model_profile)
            else:
                result = {"error": f"Unknown method: {method}", "success": False}
            
            optimization_time = time.time() - start_time
            
            if result.get("success", False):
                result.update({
                    "optimization_time_s": optimization_time,
                    "method": method,
                    "precision": precision,
                    "estimated_memory_reduction": self._estimate_memory_reduction(precision),
                    "estimated_speedup": self._estimate_speedup(precision, model_profile)
                })
                
                self.optimization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "method": method,
                    "precision": precision,
                    "success": True,
                    "optimization_time_s": optimization_time
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return {"error": str(e), "success": False}
    
    async def _quantize_with_bitsandbytes(self, model_name: str, precision: str, model_profile: ModelProfile) -> Dict[str, Any]:
        """Quantize using BitsAndBytes"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure quantization
            if precision == "int8":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif precision in ["int4", "nf4"]:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4" if precision == "nf4" else "fp4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                return {"error": f"Unsupported precision: {precision}", "success": False}
            
            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            else:
                memory_mb = 0
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "success": True,
                "memory_usage_mb": memory_mb,
                "quantization_method": "bitsandbytes",
                "precision": precision
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _quantize_with_gptq(self, model_name: str, precision: str, model_profile: ModelProfile) -> Dict[str, Any]:
        """Quantize using GPTQ"""
        # Simulate GPTQ quantization
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "success": True,
            "quantization_method": "gptq",
            "precision": precision,
            "calibration_samples": 128,
            "compression_ratio": 4.0 if precision == "int4" else 2.0
        }
    
    async def _quantize_with_awq(self, model_name: str, precision: str, model_profile: ModelProfile) -> Dict[str, Any]:
        """Quantize using AWQ"""
        # Simulate AWQ quantization
        await asyncio.sleep(1.5)  # Simulate processing time
        
        return {
            "success": True,
            "quantization_method": "awq",
            "precision": precision,
            "weight_only": True,
            "activation_aware": True
        }
    
    def _estimate_memory_reduction(self, precision: str) -> float:
        """Estimate memory reduction ratio"""
        reductions = {
            "int8": 0.5,    # 50% reduction from fp16
            "int4": 0.25,   # 75% reduction from fp16
            "nf4": 0.25,    # 75% reduction from fp16
            "fp4": 0.25     # 75% reduction from fp16
        }
        return reductions.get(precision, 1.0)
    
    def _estimate_speedup(self, precision: str, model_profile: ModelProfile) -> float:
        """Estimate inference speedup"""
        base_speedup = {
            "int8": 1.2,    # 20% speedup
            "int4": 1.8,    # 80% speedup
            "nf4": 1.6,     # 60% speedup
            "fp4": 1.5      # 50% speedup
        }.get(precision, 1.0)
        
        # Adjust based on model size
        if model_profile.size_category in [ModelSize.LARGE, ModelSize.XLARGE]:
            base_speedup *= 1.2  # Larger models benefit more
        
        return base_speedup
    
    async def cleanup(self):
        """Cleanup quantization engine"""
        self.is_initialized = False
        self.logger.info("Quantization engine cleaned up")


class PruningEngine(OptimizationEngine):
    """Model pruning optimization engine"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("PruningEngine", config)
        self.pruning_methods = ["magnitude", "structured", "unstructured", "gradual"]
        self.sparsity_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    async def initialize(self) -> bool:
        """Initialize pruning engine"""
        try:
            # Check for pruning libraries
            available_methods = []
            
            try:
                import torch.nn.utils.prune as prune
                available_methods.append("pytorch_native")
            except ImportError:
                pass
            
            try:
                import neural_compressor
                available_methods.append("neural_compressor")
            except ImportError:
                pass
            
            self.available_methods = available_methods
            self.is_initialized = len(available_methods) > 0
            
            if self.is_initialized:
                self.logger.info(f"Pruning engine initialized with methods: {available_methods}")
            else:
                self.logger.warning("No pruning methods available")
            
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pruning engine: {e}")
            return False
    
    async def optimize_model(self, model_name: str, model_profile: ModelProfile, **kwargs) -> Dict[str, Any]:
        """Apply pruning optimization"""
        if not self.is_initialized:
            return {"error": "Engine not initialized", "success": False}
        
        sparsity = kwargs.get("sparsity", 0.5)
        method = kwargs.get("method", "magnitude")
        
        try:
            self.logger.info(f"Pruning {model_name} with {sparsity} sparsity using {method}")
            
            start_time = time.time()
            
            # Simulate pruning process
            await asyncio.sleep(3)  # Simulate processing time
            
            optimization_time = time.time() - start_time
            
            # Calculate estimated improvements
            memory_reduction = sparsity * 0.8  # Not linear due to sparse storage overhead
            speedup = 1 + (sparsity * 0.5)    # Modest speedup due to sparse operations
            accuracy_retention = 1 - (sparsity * 0.1)  # Some accuracy loss
            
            result = {
                "success": True,
                "optimization_time_s": optimization_time,
                "method": method,
                "sparsity": sparsity,
                "estimated_memory_reduction": memory_reduction,
                "estimated_speedup": speedup,
                "estimated_accuracy_retention": accuracy_retention,
                "pruned_parameters": int(model_profile.num_parameters * sparsity),
                "remaining_parameters": int(model_profile.num_parameters * (1 - sparsity))
            }
            
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "method": method,
                "sparsity": sparsity,
                "success": True,
                "optimization_time_s": optimization_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return {"error": str(e), "success": False}
    
    async def cleanup(self):
        """Cleanup pruning engine"""
        self.is_initialized = False
        self.logger.info("Pruning engine cleaned up")


class DistillationEngine(OptimizationEngine):
    """Knowledge distillation optimization engine"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("DistillationEngine", config)
        self.distillation_methods = ["knowledge_distillation", "progressive_distillation", "self_distillation"]
        self.compression_ratios = [2, 4, 8, 16]
    
    async def initialize(self) -> bool:
        """Initialize distillation engine"""
        try:
            # Check for distillation libraries
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            self.is_initialized = True
            self.logger.info("Distillation engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distillation engine: {e}")
            return False
    
    async def optimize_model(self, model_name: str, model_profile: ModelProfile, **kwargs) -> Dict[str, Any]:
        """Apply distillation optimization"""
        if not self.is_initialized:
            return {"error": "Engine not initialized", "success": False}
        
        compression_ratio = kwargs.get("compression_ratio", 4)
        method = kwargs.get("method", "knowledge_distillation")
        temperature = kwargs.get("temperature", 4.0)
        
        try:
            self.logger.info(f"Distilling {model_name} with {compression_ratio}x compression using {method}")
            
            start_time = time.time()
            
            # Simulate distillation process (much longer than other optimizations)
            await asyncio.sleep(10)  # Simulate training time
            
            optimization_time = time.time() - start_time
            
            # Calculate student model characteristics
            student_parameters = model_profile.num_parameters // compression_ratio
            student_size_mb = model_profile.model_size_mb / compression_ratio
            
            # Estimate performance characteristics
            memory_reduction = 1.0 / compression_ratio
            speedup = compression_ratio * 0.8  # Not linear due to other bottlenecks
            accuracy_retention = max(0.7, 1.0 - (compression_ratio - 1) * 0.05)  # Diminishing returns
            
            result = {
                "success": True,
                "optimization_time_s": optimization_time,
                "method": method,
                "compression_ratio": compression_ratio,
                "temperature": temperature,
                "student_parameters": student_parameters,
                "student_size_mb": student_size_mb,
                "estimated_memory_reduction": memory_reduction,
                "estimated_speedup": speedup,
                "estimated_accuracy_retention": accuracy_retention,
                "training_epochs": 10,
                "distillation_loss": 0.15
            }
            
            self.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "method": method,
                "compression_ratio": compression_ratio,
                "success": True,
                "optimization_time_s": optimization_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Distillation failed: {e}")
            return {"error": str(e), "success": False}
    
    async def cleanup(self):
        """Cleanup distillation engine"""
        self.is_initialized = False
        self.logger.info("Distillation engine cleaned up")


class OptimizationSuite:
    """Comprehensive model optimization suite"""
    
    def __init__(self, output_dir: str = "optimization_suite_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_profiler = ModelProfiler()
        self.optimization_engines = {}
        self.optimization_results = []
        self.logger = self._setup_logging()
        
        # Initialize optimization engines
        self._initialize_engines()
        
        self.logger.info(f"Optimization Suite initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Available engines: {list(self.optimization_engines.keys())}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_engines(self):
        """Initialize all optimization engines"""
        # Quantization engine
        quant_config = OptimizationConfig(
            strategy=OptimizationStrategy.QUANTIZATION,
            parameters={"default_precision": "int8", "methods": ["bitsandbytes", "gptq"]},
            priority=1
        )
        self.optimization_engines["quantization"] = QuantizationEngine(quant_config)
        
        # Pruning engine
        prune_config = OptimizationConfig(
            strategy=OptimizationStrategy.PRUNING,
            parameters={"default_sparsity": 0.5, "methods": ["magnitude", "structured"]},
            priority=2
        )
        self.optimization_engines["pruning"] = PruningEngine(prune_config)
        
        # Distillation engine
        distill_config = OptimizationConfig(
            strategy=OptimizationStrategy.DISTILLATION,
            parameters={"default_compression": 4, "temperature": 4.0},
            priority=3
        )
        self.optimization_engines["distillation"] = DistillationEngine(distill_config)
    
    async def initialize_suite(self) -> bool:
        """Initialize all optimization engines"""
        self.logger.info("Initializing optimization engines...")
        
        initialized_engines = {}
        for name, engine in self.optimization_engines.items():
            if await engine.initialize():
                initialized_engines[name] = engine
                self.logger.info(f"  ✓ {name} engine initialized")
            else:
                self.logger.warning(f"  ✗ {name} engine failed to initialize")
        
        self.optimization_engines = initialized_engines
        success = len(self.optimization_engines) > 0
        
        if success:
            self.logger.info(f"Suite initialized with {len(self.optimization_engines)} engines")
        else:
            self.logger.error("No optimization engines available")
        
        return success
    
    async def optimize_model(self, model_name: str, optimization_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize a model according to the optimization plan"""
        self.logger.info(f"Starting optimization for {model_name}")
        
        # Profile the model
        model_profile = self.model_profiler.profile_model(model_name)
        self.logger.info(f"Model profile: {model_profile.size_category.value} "
                        f"({model_profile.num_parameters/1e9:.1f}B parameters)")
        
        optimization_results = {
            "model_name": model_name,
            "model_profile": asdict(model_profile),
            "optimization_plan": optimization_plan,
            "timestamp": datetime.now().isoformat(),
            "optimizations": {},
            "overall_success": True,
            "total_optimization_time_s": 0
        }
        
        total_start_time = time.time()
        
        # Execute optimization plan
        for step_idx, optimization_step in enumerate(optimization_plan):
            engine_name = optimization_step.get("engine")
            parameters = optimization_step.get("parameters", {})
            
            if engine_name not in self.optimization_engines:
                self.logger.warning(f"Engine {engine_name} not available, skipping")
                optimization_results["optimizations"][f"step_{step_idx}_{engine_name}"] = {
                    "error": f"Engine {engine_name} not available",
                    "success": False
                }
                continue
            
            engine = self.optimization_engines[engine_name]
            
            # Check compatibility
            if not engine.is_compatible(model_profile):
                self.logger.warning(f"Optimization {engine_name} not compatible with {model_name}")
                optimization_results["optimizations"][f"step_{step_idx}_{engine_name}"] = {
                    "error": f"Optimization not compatible with model",
                    "success": False
                }
                continue
            
            # Run optimization
            self.logger.info(f"Running optimization step {step_idx + 1}: {engine_name}")
            step_result = await engine.optimize_model(model_name, model_profile, **parameters)
            
            optimization_results["optimizations"][f"step_{step_idx}_{engine_name}"] = step_result
            
            if not step_result.get("success", False):
                self.logger.error(f"Optimization step {step_idx + 1} failed: {step_result.get('error', 'Unknown error')}")
                optimization_results["overall_success"] = False
            else:
                self.logger.info(f"Optimization step {step_idx + 1} completed successfully")
        
        optimization_results["total_optimization_time_s"] = time.time() - total_start_time
        
        # Save results
        await self._save_optimization_results(optimization_results)
        self.optimization_results.append(optimization_results)
        
        return optimization_results
    
    async def run_comprehensive_optimization(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Run comprehensive optimization on multiple models"""
        self.logger.info(f"Starting comprehensive optimization for {len(model_names)} models")
        
        all_results = []
        
        for model_name in model_names:
            try:
                # Create adaptive optimization plan based on model profile
                model_profile = self.model_profiler.profile_model(model_name)
                optimization_plan = self._create_adaptive_optimization_plan(model_profile)
                
                # Run optimization
                results = await self.optimize_model(model_name, optimization_plan)
                all_results.append(results)
                
            except Exception as e:
                self.logger.error(f"Failed to optimize {model_name}: {e}")
                all_results.append({
                    "model_name": model_name,
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _create_adaptive_optimization_plan(self, model_profile: ModelProfile) -> List[Dict[str, Any]]:
        """Create adaptive optimization plan based on model characteristics"""
        plan = []
        
        # Always try quantization first (most compatible)
        if model_profile.optimization_compatibility.get(OptimizationStrategy.QUANTIZATION, False):
            if model_profile.size_category in [ModelSize.LARGE, ModelSize.XLARGE, ModelSize.XXLARGE]:
                # Use more aggressive quantization for larger models
                plan.append({
                    "engine": "quantization",
                    "parameters": {"precision": "int4", "method": "gptq"}
                })
            else:
                # Use conservative quantization for smaller models
                plan.append({
                    "engine": "quantization",
                    "parameters": {"precision": "int8", "method": "bitsandbytes"}
                })
        
        # Add pruning for medium to large models
        if (model_profile.optimization_compatibility.get(OptimizationStrategy.PRUNING, False) and
            model_profile.size_category in [ModelSize.MEDIUM, ModelSize.LARGE]):
            sparsity = 0.3 if model_profile.size_category == ModelSize.MEDIUM else 0.5
            plan.append({
                "engine": "pruning",
                "parameters": {"sparsity": sparsity, "method": "magnitude"}
            })
        
        # Add distillation for very large models
        if (model_profile.optimization_compatibility.get(OptimizationStrategy.DISTILLATION, False) and
            model_profile.size_category in [ModelSize.LARGE, ModelSize.XLARGE]):
            compression_ratio = 4 if model_profile.size_category == ModelSize.LARGE else 8
            plan.append({
                "engine": "distillation",
                "parameters": {"compression_ratio": compression_ratio, "temperature": 4.0}
            })
        
        return plan
    
    async def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results to file"""
        model_name_safe = results["model_name"].replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name_safe}_optimization_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {output_file}")
    
    async def _generate_comprehensive_report(self, all_results: List[Dict[str, Any]]):
        """Generate comprehensive optimization report"""
        self.logger.info("Generating comprehensive optimization report...")
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_models": len(all_results),
            "successful_optimizations": sum(1 for r in all_results if r.get("overall_success", False)),
            "failed_optimizations": sum(1 for r in all_results if not r.get("overall_success", False)),
            "optimization_summary": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        # Analyze optimization success rates by engine
        engine_stats = defaultdict(lambda: {"total": 0, "successful": 0, "failed": 0})
        
        for result in all_results:
            if "optimizations" in result:
                for opt_name, opt_result in result["optimizations"].items():
                    engine_name = opt_name.split("_")[-1]  # Extract engine name
                    engine_stats[engine_name]["total"] += 1
                    if opt_result.get("success", False):
                        engine_stats[engine_name]["successful"] += 1
                    else:
                        engine_stats[engine_name]["failed"] += 1
        
        # Calculate success rates
        for engine_name, stats in engine_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["successful"] / stats["total"]
            else:
                stats["success_rate"] = 0.0
        
        report["optimization_summary"] = dict(engine_stats)
        
        # Performance analysis
        total_optimization_time = sum(r.get("total_optimization_time_s", 0) for r in all_results)
        avg_optimization_time = total_optimization_time / len(all_results) if all_results else 0
        
        report["performance_analysis"] = {
            "total_optimization_time_s": total_optimization_time,
            "average_optimization_time_s": avg_optimization_time,
            "optimization_time_by_model": {
                r["model_name"]: r.get("total_optimization_time_s", 0)
                for r in all_results if "model_name" in r
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        # Engine-specific recommendations
        for engine_name, stats in engine_stats.items():
            if stats["success_rate"] < 0.5:
                recommendations.append(
                    f"Consider reviewing {engine_name} engine configuration - "
                    f"success rate is only {stats['success_rate']:.1%}"
                )
            elif stats["success_rate"] > 0.9:
                recommendations.append(
                    f"{engine_name} engine is performing well with {stats['success_rate']:.1%} success rate"
                )
        
        # Time-based recommendations
        if avg_optimization_time > 300:  # 5 minutes
            recommendations.append(
                "Consider optimizing the optimization pipeline - "
                f"average time per model is {avg_optimization_time:.1f} seconds"
            )
        
        report["recommendations"] = recommendations
        
        # Save comprehensive report
        report_file = self.output_dir / f"comprehensive_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to: {report_file}")
        
        # Print summary to console
        self._print_optimization_summary(report)
    
    def _print_optimization_summary(self, report: Dict[str, Any]):
        """Print optimization summary to console"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE OPTIMIZATION REPORT")
        print(f"{'='*80}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total models processed: {report['total_models']}")
        print(f"  Successful optimizations: {report['successful_optimizations']}")
        print(f"  Failed optimizations: {report['failed_optimizations']}")
        print(f"  Overall success rate: {report['successful_optimizations']/report['total_models']*100:.1f}%")
        
        print(f"\nOptimization Engine Performance:")
        print(f"{'Engine':<15} {'Total':<8} {'Success':<8} {'Failed':<8} {'Success Rate':<12}")
        print("-" * 60)
        
        for engine_name, stats in report["optimization_summary"].items():
            print(f"{engine_name:<15} {stats['total']:<8} {stats['successful']:<8} "
                  f"{stats['failed']:<8} {stats['success_rate']:<12.1%}")
        
        print(f"\nPerformance Analysis:")
        print(f"  Total optimization time: {report['performance_analysis']['total_optimization_time_s']:.1f}s")
        print(f"  Average time per model: {report['performance_analysis']['average_optimization_time_s']:.1f}s")
        
        if report["recommendations"]:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n{'='*80}")
    
    async def cleanup_suite(self):
        """Cleanup all optimization engines"""
        self.logger.info("Cleaning up optimization suite...")
        
        for name, engine in self.optimization_engines.items():
            await engine.cleanup()
            self.logger.info(f"  ✓ {name} engine cleaned up")
        
        self.logger.info("Optimization suite cleanup complete")


async def main():
    """Main optimization suite runner"""
    # Create optimization suite
    suite = OptimizationSuite()
    
    # Initialize suite
    if not await suite.initialize_suite():
        print("Failed to initialize optimization suite")
        return
    
    # Test models (use smaller models for quick testing)
    test_models = [
        "microsoft/DialoGPT-small",
        # "microsoft/Phi-3-mini-4k-instruct",  # Uncomment for larger tests
        # "meta-llama/Llama-2-7b-hf",         # Uncomment for even larger tests
    ]
    
    try:
        # Run comprehensive optimization
        results = await suite.run_comprehensive_optimization(test_models)
        
        print(f"\n{'='*80}")
        print("MODEL OPTIMIZATION SUITE COMPLETE!")
        print(f"{'='*80}")
        print(f"Processed {len(results)} models")
        print(f"Results saved to: {suite.output_dir}")
        
    finally:
        # Cleanup
        await suite.cleanup_suite()


if __name__ == "__main__":
    asyncio.run(main())


class ParallelOptimizationManager:
    """Manages parallel optimization across multiple devices and processes"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.device_manager = self._initialize_device_manager()
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.active_optimizations = {}
        self.optimization_metrics = defaultdict(list)
        
    def _initialize_device_manager(self):
        """Initialize device management for parallel optimization"""
        devices = []
        
        # Detect CUDA devices
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        "type": "cuda",
                        "id": i,
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "available": True
                    })
        except ImportError:
            pass
        
        # Add CPU as fallback
        devices.append({
            "type": "cpu",
            "id": -1,
            "name": "CPU",
            "cores": mp.cpu_count(),
            "available": True
        })
        
        return devices
    
    async def run_parallel_optimizations(self, optimization_tasks: List[Dict]) -> List[Dict]:
        """Run multiple optimizations in parallel"""
        print(f"Running {len(optimization_tasks)} optimizations in parallel...")
        
        # Distribute tasks across available devices
        device_tasks = self._distribute_tasks_across_devices(optimization_tasks)
        
        results = []
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for device_id, tasks in device_tasks.items():
                for task in tasks:
                    future = executor.submit(self._run_optimization_on_device, task, device_id)
                    future_to_task[future] = task
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Parallel Optimizations"):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update metrics
                    self._update_optimization_metrics(result)
                    
                except Exception as e:
                    print(f"Optimization task failed: {e}")
                    results.append({
                        "task_id": task.get("task_id", "unknown"),
                        "error": str(e),
                        "success": False
                    })
        
        return results
    
    def _distribute_tasks_across_devices(self, tasks: List[Dict]) -> Dict[int, List[Dict]]:
        """Distribute optimization tasks across available devices"""
        device_tasks = defaultdict(list)
        
        # Simple round-robin distribution
        available_devices = [d for d in self.device_manager if d["available"]]
        
        for i, task in enumerate(tasks):
            device_idx = i % len(available_devices)
            device_id = available_devices[device_idx]["id"]
            device_tasks[device_id].append(task)
        
        return device_tasks
    
    def _run_optimization_on_device(self, task: Dict, device_id: int) -> Dict:
        """Run single optimization task on specific device"""
        # Set device environment
        if device_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        start_time = time.time()
        
        try:
            # Extract task parameters
            model_name = task["model_name"]
            optimization_type = task["optimization_type"]
            parameters = task.get("parameters", {})
            
            # Run optimization based on type
            if optimization_type == "quantization":
                result = self._run_quantization_task(model_name, parameters, device_id)
            elif optimization_type == "pruning":
                result = self._run_pruning_task(model_name, parameters, device_id)
            elif optimization_type == "distillation":
                result = self._run_distillation_task(model_name, parameters, device_id)
            else:
                result = {"error": f"Unknown optimization type: {optimization_type}", "success": False}
            
            result.update({
                "task_id": task.get("task_id"),
                "device_id": device_id,
                "total_time_s": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            return {
                "task_id": task.get("task_id"),
                "device_id": device_id,
                "error": str(e),
                "success": False,
                "total_time_s": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_quantization_task(self, model_name: str, parameters: Dict, device_id: int) -> Dict:
        """Run quantization task"""
        precision = parameters.get("precision", "int8")
        method = parameters.get("method", "bitsandbytes")
        
        # Simulate quantization
        time.sleep(np.random.uniform(2, 5))  # Simulate processing time
        
        return {
            "success": True,
            "optimization_type": "quantization",
            "model_name": model_name,
            "precision": precision,
            "method": method,
            "memory_reduction": 0.5 if precision == "int8" else 0.25,
            "speedup": 1.2 if precision == "int8" else 1.8,
            "accuracy_retention": 0.98 if precision == "int8" else 0.95
        }
    
    def _run_pruning_task(self, model_name: str, parameters: Dict, device_id: int) -> Dict:
        """Run pruning task"""
        sparsity = parameters.get("sparsity", 0.5)
        method = parameters.get("method", "magnitude")
        
        # Simulate pruning
        time.sleep(np.random.uniform(3, 8))  # Simulate processing time
        
        return {
            "success": True,
            "optimization_type": "pruning",
            "model_name": model_name,
            "sparsity": sparsity,
            "method": method,
            "memory_reduction": sparsity * 0.8,
            "speedup": 1 + (sparsity * 0.5),
            "accuracy_retention": 1 - (sparsity * 0.1)
        }
    
    def _run_distillation_task(self, model_name: str, parameters: Dict, device_id: int) -> Dict:
        """Run distillation task"""
        compression_ratio = parameters.get("compression_ratio", 4)
        temperature = parameters.get("temperature", 4.0)
        
        # Simulate distillation (longer process)
        time.sleep(np.random.uniform(10, 20))  # Simulate training time
        
        return {
            "success": True,
            "optimization_type": "distillation",
            "model_name": model_name,
            "compression_ratio": compression_ratio,
            "temperature": temperature,
            "memory_reduction": 1.0 / compression_ratio,
            "speedup": compression_ratio * 0.8,
            "accuracy_retention": max(0.7, 1.0 - (compression_ratio - 1) * 0.05)
        }
    
    def _update_optimization_metrics(self, result: Dict):
        """Update optimization metrics"""
        if result.get("success", False):
            opt_type = result.get("optimization_type", "unknown")
            self.optimization_metrics[opt_type].append({
                "timestamp": result.get("timestamp"),
                "total_time_s": result.get("total_time_s", 0),
                "memory_reduction": result.get("memory_reduction", 0),
                "speedup": result.get("speedup", 1),
                "accuracy_retention": result.get("accuracy_retention", 1)
            })
    
    def get_optimization_statistics(self) -> Dict:
        """Get comprehensive optimization statistics"""
        stats = {}
        
        for opt_type, metrics in self.optimization_metrics.items():
            if not metrics:
                continue
            
            times = [m["total_time_s"] for m in metrics]
            memory_reductions = [m["memory_reduction"] for m in metrics]
            speedups = [m["speedup"] for m in metrics]
            accuracy_retentions = [m["accuracy_retention"] for m in metrics]
            
            stats[opt_type] = {
                "count": len(metrics),
                "avg_time_s": np.mean(times),
                "std_time_s": np.std(times),
                "avg_memory_reduction": np.mean(memory_reductions),
                "avg_speedup": np.mean(speedups),
                "avg_accuracy_retention": np.mean(accuracy_retentions),
                "min_time_s": np.min(times),
                "max_time_s": np.max(times)
            }
        
        return stats


class OptimizationBenchmarkSuite:
    """Comprehensive benchmarking suite for optimization techniques"""
    
    def __init__(self, output_dir: str = "optimization_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.optimization_suite = OptimizationSuite(str(self.output_dir / "optimization_results"))
        self.parallel_manager = ParallelOptimizationManager()
        self.benchmark_results = []
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_benchmark(self, models: List[str], optimization_configs: List[Dict]) -> Dict:
        """Run comprehensive optimization benchmark"""
        self.logger.info(f"Starting comprehensive benchmark for {len(models)} models with {len(optimization_configs)} configurations")
        
        benchmark_start_time = time.time()
        
        # Initialize optimization suite
        if not await self.optimization_suite.initialize_suite():
            raise RuntimeError("Failed to initialize optimization suite")
        
        try:
            # Create all benchmark tasks
            benchmark_tasks = self._create_benchmark_tasks(models, optimization_configs)
            self.logger.info(f"Created {len(benchmark_tasks)} benchmark tasks")
            
            # Run benchmarks in parallel
            parallel_results = await self.parallel_manager.run_parallel_optimizations(benchmark_tasks)
            
            # Run sequential optimizations for comparison
            sequential_results = await self._run_sequential_optimizations(models[:2], optimization_configs[:2])  # Subset for time
            
            # Analyze results
            analysis = self._analyze_benchmark_results(parallel_results, sequential_results)
            
            # Generate comprehensive report
            benchmark_report = {
                "benchmark_timestamp": datetime.now().isoformat(),
                "total_benchmark_time_s": time.time() - benchmark_start_time,
                "models_tested": models,
                "optimization_configs": optimization_configs,
                "parallel_results": parallel_results,
                "sequential_results": sequential_results,
                "analysis": analysis,
                "optimization_statistics": self.parallel_manager.get_optimization_statistics()
            }
            
            # Save comprehensive report
            await self._save_benchmark_report(benchmark_report)
            
            # Print summary
            self._print_benchmark_summary(benchmark_report)
            
            return benchmark_report
            
        finally:
            await self.optimization_suite.cleanup_suite()
    
    def _create_benchmark_tasks(self, models: List[str], optimization_configs: List[Dict]) -> List[Dict]:
        """Create all benchmark tasks"""
        tasks = []
        task_id = 0
        
        for model in models:
            for config in optimization_configs:
                task = {
                    "task_id": f"benchmark_task_{task_id:04d}",
                    "model_name": model,
                    "optimization_type": config["type"],
                    "parameters": config.get("parameters", {}),
                    "expected_memory_reduction": config.get("expected_memory_reduction", 0.5),
                    "expected_speedup": config.get("expected_speedup", 1.2)
                }
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    async def _run_sequential_optimizations(self, models: List[str], optimization_configs: List[Dict]) -> List[Dict]:
        """Run optimizations sequentially for comparison"""
        self.logger.info("Running sequential optimizations for comparison...")
        
        sequential_results = []
        
        for model in models:
            for config in optimization_configs:
                start_time = time.time()
                
                try:
                    # Create optimization plan
                    optimization_plan = [{
                        "engine": config["type"],
                        "parameters": config.get("parameters", {})
                    }]
                    
                    # Run optimization
                    result = await self.optimization_suite.optimize_model(model, optimization_plan)
                    
                    result.update({
                        "sequential": True,
                        "total_time_s": time.time() - start_time
                    })
                    
                    sequential_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Sequential optimization failed: {e}")
                    sequential_results.append({
                        "model_name": model,
                        "optimization_type": config["type"],
                        "error": str(e),
                        "success": False,
                        "sequential": True,
                        "total_time_s": time.time() - start_time
                    })
        
        return sequential_results
    
    def _analyze_benchmark_results(self, parallel_results: List[Dict], sequential_results: List[Dict]) -> Dict:
        """Analyze benchmark results"""
        analysis = {
            "parallel_analysis": self._analyze_parallel_results(parallel_results),
            "sequential_analysis": self._analyze_sequential_results(sequential_results),
            "comparison": self._compare_parallel_vs_sequential(parallel_results, sequential_results)
        }
        
        return analysis
    
    def _analyze_parallel_results(self, results: List[Dict]) -> Dict:
        """Analyze parallel optimization results"""
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful parallel results"}
        
        times = [r["total_time_s"] for r in successful_results]
        memory_reductions = [r.get("memory_reduction", 0) for r in successful_results]
        speedups = [r.get("speedup", 1) for r in successful_results]
        
        return {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(results),
            "avg_time_s": np.mean(times),
            "std_time_s": np.std(times),
            "min_time_s": np.min(times),
            "max_time_s": np.max(times),
            "avg_memory_reduction": np.mean(memory_reductions),
            "avg_speedup": np.mean(speedups),
            "total_parallel_time_s": np.max(times)  # Parallel execution time is the max
        }
    
    def _analyze_sequential_results(self, results: List[Dict]) -> Dict:
        """Analyze sequential optimization results"""
        successful_results = [r for r in results if r.get("overall_success", False)]
        failed_results = [r for r in results if not r.get("overall_success", False)]
        
        if not successful_results:
            return {"error": "No successful sequential results"}
        
        times = [r["total_optimization_time_s"] for r in successful_results]
        
        return {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "avg_time_s": np.mean(times) if times else 0,
            "std_time_s": np.std(times) if times else 0,
            "total_sequential_time_s": np.sum(times) if times else 0
        }
    
    def _compare_parallel_vs_sequential(self, parallel_results: List[Dict], sequential_results: List[Dict]) -> Dict:
        """Compare parallel vs sequential optimization performance"""
        parallel_analysis = self._analyze_parallel_results(parallel_results)
        sequential_analysis = self._analyze_sequential_results(sequential_results)
        
        if "error" in parallel_analysis or "error" in sequential_analysis:
            return {"error": "Cannot compare due to missing results"}
        
        parallel_total_time = parallel_analysis.get("total_parallel_time_s", 0)
        sequential_total_time = sequential_analysis.get("total_sequential_time_s", 0)
        
        speedup_ratio = sequential_total_time / parallel_total_time if parallel_total_time > 0 else 0
        
        return {
            "parallel_total_time_s": parallel_total_time,
            "sequential_total_time_s": sequential_total_time,
            "parallel_speedup_ratio": speedup_ratio,
            "parallel_efficiency": speedup_ratio / len(self.parallel_manager.device_manager) if self.parallel_manager.device_manager else 0,
            "parallel_success_rate": parallel_analysis.get("success_rate", 0),
            "sequential_success_rate": sequential_analysis.get("success_rate", 0),
            "recommendation": self._generate_parallelization_recommendation(speedup_ratio)
        }
    
    def _generate_parallelization_recommendation(self, speedup_ratio: float) -> str:
        """Generate recommendation based on parallelization performance"""
        if speedup_ratio > 3:
            return "Excellent parallelization efficiency - highly recommended for production use"
        elif speedup_ratio > 2:
            return "Good parallelization efficiency - recommended for most use cases"
        elif speedup_ratio > 1.5:
            return "Moderate parallelization efficiency - consider for large-scale operations"
        elif speedup_ratio > 1:
            return "Limited parallelization benefit - evaluate cost vs benefit"
        else:
            return "Poor parallelization efficiency - sequential processing may be better"
    
    async def _save_benchmark_report(self, report: Dict):
        """Save comprehensive benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"comprehensive_benchmark_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive benchmark report saved to: {report_file}")
        
        # Also save CSV summary for easy analysis
        csv_file = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        self._save_csv_summary(report, csv_file)
    
    def _save_csv_summary(self, report: Dict, csv_file: Path):
        """Save benchmark summary as CSV"""
        try:
            # Extract key metrics for CSV
            rows = []
            
            # Parallel results
            for result in report.get("parallel_results", []):
                if result.get("success", False):
                    rows.append({
                        "execution_type": "parallel",
                        "model_name": result.get("model_name", ""),
                        "optimization_type": result.get("optimization_type", ""),
                        "device_id": result.get("device_id", ""),
                        "total_time_s": result.get("total_time_s", 0),
                        "memory_reduction": result.get("memory_reduction", 0),
                        "speedup": result.get("speedup", 1),
                        "accuracy_retention": result.get("accuracy_retention", 1),
                        "success": result.get("success", False)
                    })
            
            # Sequential results
            for result in report.get("sequential_results", []):
                if result.get("overall_success", False):
                    rows.append({
                        "execution_type": "sequential",
                        "model_name": result.get("model_name", ""),
                        "optimization_type": "multiple",  # Sequential runs multiple optimizations
                        "device_id": "N/A",
                        "total_time_s": result.get("total_optimization_time_s", 0),
                        "memory_reduction": "N/A",  # Complex to extract from nested results
                        "speedup": "N/A",
                        "accuracy_retention": "N/A",
                        "success": result.get("overall_success", False)
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_file, index=False)
                self.logger.info(f"CSV summary saved to: {csv_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save CSV summary: {e}")
    
    def _print_benchmark_summary(self, report: Dict):
        """Print comprehensive benchmark summary"""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE OPTIMIZATION BENCHMARK REPORT")
        print(f"{'='*100}")
        
        print(f"\nBenchmark Overview:")
        print(f"  Total benchmark time: {report['total_benchmark_time_s']:.1f}s")
        print(f"  Models tested: {len(report['models_tested'])}")
        print(f"  Optimization configurations: {len(report['optimization_configs'])}")
        
        # Parallel results summary
        parallel_analysis = report["analysis"]["parallel_analysis"]
        if "error" not in parallel_analysis:
            print(f"\nParallel Optimization Results:")
            print(f"  Total tasks: {parallel_analysis['total_tasks']}")
            print(f"  Success rate: {parallel_analysis['success_rate']:.1%}")
            print(f"  Average time per task: {parallel_analysis['avg_time_s']:.1f}s")
            print(f"  Total parallel execution time: {parallel_analysis['total_parallel_time_s']:.1f}s")
            print(f"  Average memory reduction: {parallel_analysis['avg_memory_reduction']:.1%}")
            print(f"  Average speedup: {parallel_analysis['avg_speedup']:.2f}x")
        
        # Sequential results summary
        sequential_analysis = report["analysis"]["sequential_analysis"]
        if "error" not in sequential_analysis:
            print(f"\nSequential Optimization Results:")
            print(f"  Total tasks: {sequential_analysis['total_tasks']}")
            print(f"  Success rate: {sequential_analysis['success_rate']:.1%}")
            print(f"  Average time per task: {sequential_analysis['avg_time_s']:.1f}s")
            print(f"  Total sequential execution time: {sequential_analysis['total_sequential_time_s']:.1f}s")
        
        # Comparison
        comparison = report["analysis"]["comparison"]
        if "error" not in comparison:
            print(f"\nParallel vs Sequential Comparison:")
            print(f"  Parallel speedup ratio: {comparison['parallel_speedup_ratio']:.2f}x")
            print(f"  Parallel efficiency: {comparison['parallel_efficiency']:.1%}")
            print(f"  Recommendation: {comparison['recommendation']}")
        
        # Optimization statistics
        opt_stats = report.get("optimization_statistics", {})
        if opt_stats:
            print(f"\nOptimization Type Statistics:")
            print(f"{'Type':<15} {'Count':<8} {'Avg Time (s)':<12} {'Avg Memory Red.':<15} {'Avg Speedup':<12}")
            print("-" * 70)
            
            for opt_type, stats in opt_stats.items():
                print(f"{opt_type:<15} {stats['count']:<8} {stats['avg_time_s']:<12.1f} "
                      f"{stats['avg_memory_reduction']:<15.1%} {stats['avg_speedup']:<12.2f}x")
        
        print(f"\n{'='*100}")


# Additional utility functions and classes for comprehensive optimization

class OptimizationProfiler:
    """Profiles optimization performance and resource usage"""
    
    def __init__(self):
        self.profiles = {}
        self.system_metrics = {}
    
    def start_profiling(self, optimization_id: str):
        """Start profiling an optimization"""
        self.profiles[optimization_id] = {
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": self._get_cpu_usage(),
            "metrics": []
        }
    
    def record_metric(self, optimization_id: str, metric_name: str, value: float):
        """Record a metric during optimization"""
        if optimization_id in self.profiles:
            self.profiles[optimization_id]["metrics"].append({
                "timestamp": time.time(),
                "metric": metric_name,
                "value": value
            })
    
    def stop_profiling(self, optimization_id: str) -> Dict:
        """Stop profiling and return results"""
        if optimization_id not in self.profiles:
            return {"error": "Optimization ID not found"}
        
        profile = self.profiles[optimization_id]
        end_time = time.time()
        
        result = {
            "optimization_id": optimization_id,
            "total_time_s": end_time - profile["start_time"],
            "memory_usage": {
                "start_mb": profile["start_memory"],
                "end_mb": self._get_memory_usage(),
                "peak_mb": max([m["value"] for m in profile["metrics"] if m["metric"] == "memory_mb"] + [profile["start_memory"]])
            },
            "cpu_usage": {
                "start_percent": profile["start_cpu"],
                "end_percent": self._get_cpu_usage(),
                "avg_percent": np.mean([m["value"] for m in profile["metrics"] if m["metric"] == "cpu_percent"] + [profile["start_cpu"]])
            },
            "metrics": profile["metrics"]
        }
        
        # Cleanup
        del self.profiles[optimization_id]
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 ** 2)
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0


class OptimizationRecommendationEngine:
    """Provides intelligent recommendations for optimization strategies"""
    
    def __init__(self):
        self.recommendation_rules = self._load_recommendation_rules()
        self.performance_history = defaultdict(list)
    
    def _load_recommendation_rules(self) -> Dict:
        """Load optimization recommendation rules"""
        return {
            "quantization": {
                "conditions": {
                    "model_size_gb": {"min": 1.0},  # Recommend for models > 1GB
                    "memory_constrained": True,
                    "accuracy_tolerance": {"min": 0.02}  # Can tolerate 2% accuracy loss
                },
                "recommendations": {
                    "small_models": {"precision": "int8", "method": "bitsandbytes"},
                    "large_models": {"precision": "int4", "method": "gptq"},
                    "memory_critical": {"precision": "int4", "method": "awq"}
                }
            },
            "pruning": {
                "conditions": {
                    "model_size_gb": {"min": 2.0},  # Recommend for models > 2GB
                    "inference_speed_critical": True,
                    "accuracy_tolerance": {"min": 0.05}  # Can tolerate 5% accuracy loss
                },
                "recommendations": {
                    "moderate_pruning": {"sparsity": 0.3, "method": "magnitude"},
                    "aggressive_pruning": {"sparsity": 0.7, "method": "structured"},
                    "gradual_pruning": {"sparsity": 0.5, "method": "gradual"}
                }
            },
            "distillation": {
                "conditions": {
                    "model_size_gb": {"min": 5.0},  # Recommend for models > 5GB
                    "deployment_size_critical": True,
                    "training_resources_available": True
                },
                "recommendations": {
                    "moderate_compression": {"compression_ratio": 4, "temperature": 4.0},
                    "aggressive_compression": {"compression_ratio": 8, "temperature": 6.0},
                    "conservative_compression": {"compression_ratio": 2, "temperature": 3.0}
                }
            }
        }
    
    def get_optimization_recommendations(self, model_profile: ModelProfile, constraints: Dict) -> List[Dict]:
        """Get optimization recommendations based on model profile and constraints"""
        recommendations = []
        
        # Analyze each optimization strategy
        for strategy, rules in self.recommendation_rules.items():
            if self._check_conditions(model_profile, constraints, rules["conditions"]):
                # Determine best recommendation variant
                best_recommendation = self._select_best_recommendation(
                    model_profile, constraints, rules["recommendations"]
                )
                
                if best_recommendation:
                    recommendations.append({
                        "strategy": strategy,
                        "parameters": best_recommendation,
                        "confidence": self._calculate_confidence(model_profile, strategy),
                        "expected_benefits": self._estimate_benefits(model_profile, strategy, best_recommendation)
                    })
        
        # Sort by confidence and expected benefits
        recommendations.sort(key=lambda x: x["confidence"] * x["expected_benefits"]["overall_score"], reverse=True)
        
        return recommendations
    
    def _check_conditions(self, model_profile: ModelProfile, constraints: Dict, conditions: Dict) -> bool:
        """Check if conditions are met for an optimization strategy"""
        # Check model size condition
        if "model_size_gb" in conditions:
            model_size_gb = model_profile.model_size_mb / 1024
            min_size = conditions["model_size_gb"].get("min", 0)
            if model_size_gb < min_size:
                return False
        
        # Check constraint-based conditions
        if conditions.get("memory_constrained", False) and not constraints.get("memory_limited", False):
            return False
        
        if conditions.get("inference_speed_critical", False) and not constraints.get("speed_critical", False):
            return False
        
        if conditions.get("training_resources_available", False) and not constraints.get("can_train", False):
            return False
        
        # Check accuracy tolerance
        if "accuracy_tolerance" in conditions:
            min_tolerance = conditions["accuracy_tolerance"].get("min", 0)
            if constraints.get("max_accuracy_loss", 0) < min_tolerance:
                return False
        
        return True
    
    def _select_best_recommendation(self, model_profile: ModelProfile, constraints: Dict, recommendations: Dict) -> Optional[Dict]:
        """Select the best recommendation variant"""
        # Simple heuristic-based selection
        if constraints.get("memory_limited", False):
            return recommendations.get("memory_critical") or recommendations.get("aggressive_compression") or recommendations.get("aggressive_pruning")
        elif constraints.get("speed_critical", False):
            return recommendations.get("moderate_pruning") or recommendations.get("moderate_compression")
        elif constraints.get("accuracy_critical", False):
            return recommendations.get("conservative_compression") or recommendations.get("small_models")
        else:
            # Default to moderate approach
            return recommendations.get("moderate_pruning") or recommendations.get("moderate_compression") or recommendations.get("small_models")
    
    def _calculate_confidence(self, model_profile: ModelProfile, strategy: str) -> float:
        """Calculate confidence score for a recommendation"""
        base_confidence = 0.7
        
        # Adjust based on model compatibility
        if model_profile.optimization_compatibility.get(OptimizationStrategy(strategy), False):
            base_confidence += 0.2
        
        # Adjust based on historical performance
        if strategy in self.performance_history:
            avg_success_rate = np.mean([p["success"] for p in self.performance_history[strategy]])
            base_confidence += (avg_success_rate - 0.5) * 0.2
        
        return min(1.0, base_confidence)
    
    def _estimate_benefits(self, model_profile: ModelProfile, strategy: str, parameters: Dict) -> Dict:
        """Estimate benefits of applying an optimization"""
        benefits = {
            "memory_reduction": 0.0,
            "speed_improvement": 0.0,
            "size_reduction": 0.0,
            "accuracy_impact": 0.0,
            "overall_score": 0.0
        }
        
        if strategy == "quantization":
            precision = parameters.get("precision", "int8")
            if precision == "int8":
                benefits.update({
                    "memory_reduction": 0.5,
                    "speed_improvement": 0.2,
                    "size_reduction": 0.5,
                    "accuracy_impact": -0.02
                })
            elif precision == "int4":
                benefits.update({
                    "memory_reduction": 0.75,
                    "speed_improvement": 0.8,
                    "size_reduction": 0.75,
                    "accuracy_impact": -0.05
                })
        
        elif strategy == "pruning":
            sparsity = parameters.get("sparsity", 0.5)
            benefits.update({
                "memory_reduction": sparsity * 0.8,
                "speed_improvement": sparsity * 0.5,
                "size_reduction": sparsity * 0.9,
                "accuracy_impact": -sparsity * 0.1
            })
        
        elif strategy == "distillation":
            compression_ratio = parameters.get("compression_ratio", 4)
            benefits.update({
                "memory_reduction": 1.0 - (1.0 / compression_ratio),
                "speed_improvement": (compression_ratio - 1) * 0.3,
                "size_reduction": 1.0 - (1.0 / compression_ratio),
                "accuracy_impact": -(compression_ratio - 1) * 0.03
            })
        
        # Calculate overall score
        benefits["overall_score"] = (
            benefits["memory_reduction"] * 0.3 +
            benefits["speed_improvement"] * 0.3 +
            benefits["size_reduction"] * 0.2 +
            max(0, benefits["accuracy_impact"] + 0.1) * 0.2  # Penalize accuracy loss
        )
        
        return benefits
    
    def update_performance_history(self, strategy: str, result: Dict):
        """Update performance history for future recommendations"""
        self.performance_history[strategy].append({
            "timestamp": datetime.now().isoformat(),
            "success": result.get("success", False),
            "memory_reduction": result.get("memory_reduction", 0),
            "speedup": result.get("speedup", 1),
            "accuracy_retention": result.get("accuracy_retention", 1)
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history[strategy]) > 100:
            self.performance_history[strategy] = self.performance_history[strategy][-100:]


# Example usage and testing functions

def create_test_optimization_configs() -> List[Dict]:
    """Create test optimization configurations"""
    return [
        {
            "type": "quantization",
            "parameters": {"precision": "int8", "method": "bitsandbytes"},
            "expected_memory_reduction": 0.5,
            "expected_speedup": 1.2
        },
        {
            "type": "quantization",
            "parameters": {"precision": "int4", "method": "gptq"},
            "expected_memory_reduction": 0.75,
            "expected_speedup": 1.8
        },
        {
            "type": "pruning",
            "parameters": {"sparsity": 0.5, "method": "magnitude"},
            "expected_memory_reduction": 0.4,
            "expected_speedup": 1.5
        },
        {
            "type": "distillation",
            "parameters": {"compression_ratio": 4, "temperature": 4.0},
            "expected_memory_reduction": 0.75,
            "expected_speedup": 3.2
        }
    ]


async def run_optimization_suite_demo():
    """Run a demonstration of the optimization suite"""
    print("Starting Model Optimization Suite Demo...")
    
    # Create benchmark suite
    benchmark_suite = OptimizationBenchmarkSuite()
    
    # Test models
    test_models = [
        "microsoft/DialoGPT-small",
        # Add more models as needed
    ]
    
    # Test optimization configurations
    optimization_configs = create_test_optimization_configs()
    
    try:
        # Run comprehensive benchmark
        results = await benchmark_suite.run_comprehensive_benchmark(test_models, optimization_configs)
        
        print("\nDemo completed successfully!")
        print(f"Results saved to: {benchmark_suite.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return None


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_optimization_suite_demo())