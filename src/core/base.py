"""
Base classes and interfaces for the evaluation pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel
from pathlib import Path
import json

class BaseDataset(BaseModel):
    """Base model for dataset configuration"""
    name: str
    path: str | List[str] | Path | List[Path]
    evaluator: str
    shuffle: bool = False
    description: Optional[str] = None
    repeat_num: int = 1  # Number of times to repeat the dataset


class BaseModelConfig(BaseModel):
    """Base model for LLM configuration"""
    name: str
    urls: str | List[str]
    per_url_max_workers: int
    api_key: str
    api_type: str = "default"
    api_version: str = "default"
    model: str = "default"
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    timeout: int = 30
    max_tokens: int = 8192
    system_prompt: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = {}
    streaming: bool = False  # Whether to use streaming response
    use_chat: bool = True   # Whether to use chat completion or text completion


class EvaluationResult(BaseModel):
    """Model for evaluation results - supports flexible metrics"""
    dataset_name: str
    model_name: str
    timestamp: str
    # Main metrics - flexible metrics dict, support nested structure
    metrics: Optional[Union[float, int, Dict[str, Any]]] = None
    # Sample statistics
    sample_stats: Optional[Dict[str, int]] = None
    # Details and raw results
    details: Optional[Dict[str, Any]] = None

    # Backward compatibility properties
    @property
    def accuracy(self) -> float:
        """Backward compatibility: return accuracy metric"""
        if isinstance(self.metrics, dict):
            # Try to get accuracy from nested dict
            if 'accuracy' in self.metrics:
                value = self.metrics['accuracy']
                return float(value) if isinstance(value, (int, float)) else 0.0
            # Try to get accuracy from first task
            for task_metrics in self.metrics.values():
                if isinstance(task_metrics, dict) and 'accuracy' in task_metrics:
                    return float(task_metrics['accuracy'])
        elif isinstance(self.metrics, (int, float)):
            return float(self.metrics)
        return 0.0
    
    @property 
    def total_samples(self) -> int:
        """Backward compatibility: return total sample count"""
        if isinstance(self.metrics, dict):
            # Try to get total field
            if 'total' in self.metrics:
                return int(self.metrics['total'])
            # Get from sample_stats
            if self.sample_stats and 'total' in self.sample_stats:
                return self.sample_stats['total']
        return 0
    
    @property
    def correct_samples(self) -> int:
        """Backward compatibility: return correct sample count"""
        if self.sample_stats and 'correct' in self.sample_stats:
            return self.sample_stats['correct']
        return 0
    
    def get_primary_metric(self) -> "Tuple[str, Union[float, int, str]]":
        """Get primary metric (first metric or accuracy)"""
        if isinstance(self.metrics, dict):
            if 'accuracy' in self.metrics:
                return 'accuracy', self.metrics['accuracy']
            elif self.metrics:
                first_key = next(iter(self.metrics))
                return first_key, self.metrics[first_key]
        elif isinstance(self.metrics, (int, float)):
            return 'score', self.metrics
        return 'unknown', 0.0


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    @abstractmethod
    def load(self, dataset_config: BaseDataset) -> List[Dict[str, Any]]:
        """Load dataset and return list of samples"""
        pass
    
    @abstractmethod
    def validate(self, dataset_config: BaseDataset) -> bool:
        """Validate dataset configuration"""
        pass


class BaseModelClient(ABC):
    """Abstract base class for model clients"""
    
    @abstractmethod
    async def generate_batch(self, prompt: str, **kwargs) -> str:
        """Generate response from model"""
        pass
    
    @abstractmethod
    def validate_config(self, config: BaseModelConfig) -> bool:
        """Validate model configuration"""
        pass


class BaseEvaluator(ABC):

    def __init__(self, dataset_config: Optional[BaseDataset] = None, **kwargs):
        self.dataset_config = dataset_config
        self.kwargs = kwargs

    """Abstract base class for evaluators"""
    def extract_format_prompt(self, dataset_config: BaseDataset, sample: Dict[str, Any]) -> str:
        """Format prompt"""
        prompt = self._extract_prompt(sample)
        return prompt
    
    @abstractmethod
    def evaluate(self, predictions: List[str], ground_truths: List[str], **kwargs) -> Dict[str, float]:
        """Evaluate predictions against ground truths"""
        pass

    def _extract_prompt(self, sample: Dict[str, Any]) -> str:
        """Format sample into prompt for JSON datasets with case-insensitive matching"""
        # Define candidate fields (lowercase)
        target_fields = [
            "question", "prompt", "input", "problem",
            "text", "content", "query"
        ]

        # Create case-insensitive lookup
        sample_keys_lower = {k.lower(): k for k in sample.keys()}
        
        for field in target_fields:
            if field in sample_keys_lower:
                original_key = sample_keys_lower[field]
                candidate = sample[original_key]
                if candidate and isinstance(candidate, str) and candidate.strip():
                    return candidate
        
        raise ValueError(f"Cannot find prompt field in sample. Available keys: {list(sample.keys())}")

    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Extract ground truth with case-insensitive matching"""
        target_fields = ["answer", "target", "output", "label", "solution"]
        sample_keys_lower = {k.lower(): k for k in sample.keys()}
        
        for field in target_fields:
            if field in sample_keys_lower:
                original_key = sample_keys_lower[field]
                candidate = sample[original_key]
                
                if candidate is not None:
                    if isinstance(candidate, str):
                        return candidate.strip()
                    else:
                        return json.dumps(candidate)
                        
        print(sample["output"])
        raise ValueError(f"Cannot find ground truth field in sample. Available keys: {list(sample.keys())}, target_fields: {target_fields}")
    