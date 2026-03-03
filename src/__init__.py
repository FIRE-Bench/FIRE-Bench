"""
LLM Evaluation Pipeline

A flexible and extensible pipeline for evaluating Large Language Models
on various benchmarks and datasets.
"""

__version__ = "1.0.0"
__author__ = "FIRE Evaluation"

from .core import EvaluationPipeline, OpenAIModelClient, ModelClientFactory, DatasetLoader
from .utils import ConfigManager

__all__ = [
    "EvaluationPipeline",
    "OpenAIModelClient",
    "ModelClientFactory",
    "DatasetLoader", 
    "ConfigManager",
] 