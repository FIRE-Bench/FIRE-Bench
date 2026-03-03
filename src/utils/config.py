"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from ..core.base import BaseDataset, BaseModelConfig
from .path_manager import get_config_path, resolve_dataset_path, get_project_root


class ConfigManager:
    """Configuration manager for datasets and models"""
    
    def __init__(self, config_file: str = "config/datasets.yaml"):
        self.config_file = get_config_path(config_file)
        self.datasets_config = self._load_datasets_config()
    
    def _load_datasets_config(self) -> Dict[str, Any]:
        """Load datasets configuration from YAML"""
        if not self.config_file.exists():
            logger.warning(f"Datasets config file not found: {self.config_file}")
            return {"datasets": {}, "defaults": {}}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded datasets config from: {self.config_file}")
                return config
        except Exception as e:
            logger.error(f"Error loading datasets config: {str(e)}")
            return {"datasets": {}, "defaults": {}}
    
    def get_dataset_config(self, dataset_name: str) -> Optional[BaseDataset]:
        """Get configuration for a specific dataset"""
        if dataset_name not in self.datasets_config["datasets"]:
            logger.error(f"Dataset '{dataset_name}' not found in configuration")
            return None
        
        config_data = self.datasets_config["datasets"][dataset_name]
        
        try:
            # Get dataset path
            path = config_data.get("path") or config_data.get("dataset_path")
            if not path:
                logger.error(f"No path specified for dataset '{dataset_name}'")
                return None

            # Use path manager to resolve dataset paths
            resolved_paths = resolve_dataset_path(path)

            return BaseDataset(
                name=config_data["name"],
                path=resolved_paths,  # Convert to string for compatibility
                shuffle=config_data.get("shuffle", False),
                description=config_data.get("description"),
                evaluator=config_data.get("evaluator", None),
                repeat_num=config_data.get("repeat_num", 1),  # Default value is 1
            )
        except Exception as e:
            logger.error(f"Error creating dataset config for '{dataset_name}': {str(e)}")
            return None
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self.datasets_config["datasets"].keys())
    
    def get_datasets_by_category(self, category: str) -> List[str]:
        """Get datasets by category"""
        datasets = []
        for name, config in self.datasets_config["datasets"].items():
            if config.get("category") == category:
                datasets.append(name)
        return datasets
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default evaluation settings"""
        return self.datasets_config.get("defaults", {})
    
    def create_model_config(self, 
                          name: str,
                          urls: List[str],  # Changed from url to urls (list)
                          api_key: str,
                          api_type: str = "",
                          model: str = "",
                          per_url_max_workers: int = 128,
                          **kwargs) -> BaseModelConfig:
        """Create model configuration with multiple URLs support"""
        defaults = self.get_default_settings()
        
        config = BaseModelConfig(
            name=name,
            urls=urls,
            per_url_max_workers=per_url_max_workers,
            api_key=api_key,
            model=model,
            api_type=api_type,
            api_version=kwargs.get("api_version", defaults.get("api_version", "2023-05-15")),
            temperature=kwargs.get("temperature", defaults.get("temperature", 0.0)),
            top_p=kwargs.get("top_p", defaults.get("top_p", None)),
            top_k=kwargs.get("top_k", defaults.get("top_k", None)),
            max_tokens=kwargs.get("max_tokens", defaults.get("max_tokens", 1024)),
            timeout=kwargs.get("timeout", defaults.get("timeout", 30)),
            system_prompt=kwargs.get("system_prompt", defaults.get("system_prompt", None)),
            extra_body=kwargs.get("extra_body", defaults.get("extra_body", {})),
            streaming=kwargs.get("streaming", False),
            use_chat=kwargs.get("use_chat", True)
        )
        
        return config
    
    def validate_datasets(self, dataset_names: List[str]) -> List[str]:
        """Validate that all dataset names exist"""
        available_datasets = self.get_available_datasets()
        invalid_datasets = [name for name in dataset_names if name not in available_datasets]
        
        if invalid_datasets:
            logger.error(f"Invalid datasets: {invalid_datasets}")
            logger.info(f"Available datasets: {available_datasets}")
        
        return invalid_datasets
    
    def list_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all datasets"""
        info = {}
        for name, config in self.datasets_config["datasets"].items():
            # Resolve dataset path to get full path information
            dataset_path = config.get("path", "")
            resolved_path = resolve_dataset_path(dataset_path) if dataset_path else ""
            
            dataset_info = {
                "name": config["name"],
                "description": config.get("description", "No description"),
                "category": config.get("category", "unknown"),
                "path": dataset_path,
                "resolved_path": [str(path) for path in resolved_path],
                "exists": all(path.exists() for path in resolved_path) if resolved_path else False
            }
            
            info[name] = dataset_info
        return info