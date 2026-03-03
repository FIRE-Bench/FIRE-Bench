"""
Evaluators for different types of datasets
"""

from typing import Dict, List, Any, Optional
from loguru import logger
from ..base import BaseEvaluator, BaseDataset
from src.utils.path_manager import get_project_root
ROOT_DIR = get_project_root()



class EvaluatorManager:
    """Main evaluator class"""
    
    def __init__(self):
        self.factory = {}

    def register(self, key: str):
        def decorator(evaluator_class):
            self.factory[key] = evaluator_class
            return evaluator_class
        return decorator

    def build(self, key: str, **kwargs)->BaseEvaluator:
        if key not in self.factory:
            raise ValueError(f"Evaluator {key} not found, supported evaluators: {list(self.factory.keys())}")
        return self.factory[key](**kwargs)

evaluator_manager = EvaluatorManager()


@evaluator_manager.register("fire")
class FIREMCQEvaluatorAdapter(BaseEvaluator):
    """Adapter for FIRE evaluator to work with the pipeline"""
    
    def __init__(self, dataset_config: Optional[BaseDataset] = None, demo_count: int = 0, **kwargs):
        from .fire_evaluator import FIREMCQEvaluator
        self.fire_mcq_evaluator = FIREMCQEvaluator(dataset_config, demo_count=demo_count, **kwargs)
        self.dataset_config = dataset_config
        self.demo_count = demo_count
    
    def extract_format_prompt(self, dataset_config: BaseDataset, sample: Dict[str, Any]) -> str:
        """Extract format prompt"""
        return self.fire_mcq_evaluator.extract_format_prompt(dataset_config, sample)
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Extract ground truth"""
        return self.fire_mcq_evaluator.extract_ground_truth(sample)
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], 
                 data_samples: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, float]:
        """Evaluate using FIRE evaluator"""
        return self.fire_mcq_evaluator.evaluate(predictions, ground_truths, data_samples, **kwargs)
    

@evaluator_manager.register("fire_scene")
class FireSceneEvaluatorAdapter(BaseEvaluator):
    def __init__(self, dataset_config: Optional[BaseDataset] = None, **kwargs):
        from .fire_scene_evaluator import FireSceneEvaluator
        self.evaluator = FireSceneEvaluator(dataset_config, **kwargs)
        self.dataset_config = dataset_config

    def extract_format_prompt(self, dataset_config: BaseDataset, sample: Dict[str, Any]) -> str:
        return self.evaluator.extract_format_prompt(dataset_config, sample)
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        return self.evaluator.extract_ground_truth(sample)
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], **kwargs) -> Dict[str, float]:
        return self.evaluator.evaluate(predictions, ground_truths, **kwargs)
    
    async def evaluate_async(self, predictions, ground_truths, **kwargs):
        return await self.evaluator.evaluate_async(predictions, ground_truths, **kwargs)

