"""
Project path management module.
Automatically finds project root directory using pyrootutils and provides unified path management.
"""

import os
from pathlib import Path
from typing import Union, Optional, List
import pyrootutils

class ProjectPathManager:
    """Project path manager singleton."""
    
    _instance = None
    _project_root = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._setup_project_root()
        return cls._instance
    
    @classmethod
    def _setup_project_root(cls):
        """Setup project root directory."""
        try:
            root = pyrootutils.find_root(
                search_from=__file__,
                indicator=[".gitignore", "pyproject.toml", "setup.py", "requirements.txt"]
            )
            if root is not None:
                cls._project_root = root
                print(f"Project root found: {cls._project_root}")
            else:
                raise Exception("Could not find project root")
        except Exception as e:
            current_file = Path(__file__).resolve()
            cls._project_root = current_file.parent.parent.parent
            print(f"Project root fallback: {cls._project_root}")
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        if self._project_root is None:
            raise RuntimeError("Project root not initialized")
        return self._project_root
    
    def get_path(self, relative_path: Union[str, Path]) -> Path:
        """Get absolute path relative to project root."""
        if self._project_root is None:
            raise RuntimeError("Project root not initialized")
        return self._project_root / relative_path
    
    def get_dataset_path(self, dataset_path: Union[str, Path]) -> Path:
        """Get dataset path."""
        return self.get_path(dataset_path)
    
    def get_config_path(self, config_file: str = "config/datasets.yaml") -> Path:
        """Get config file path."""
        return self.get_path(config_file)
    
    def get_results_path(self, results_dir: str = "results") -> Path:
        """Get results directory path."""
        path = self.get_path(results_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_path(self, results_dir: Union[str, Path]) -> Path:
        """Get cache directory path."""
        path = self.get_path(results_dir) / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_irm_cache_path(self, results_dir: Union[str, Path]) -> Path:
        """Get IRM cache directory path."""
        path = self.get_path(results_dir) / "irm_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def ensure_path_exists(self, path: Union[str, Path]) -> Path:
        """Ensure path exists, create if not exists."""
        full_path = self.get_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path
    
    def path_exists(self, relative_path: Union[str, Path]) -> bool:
        """Check if relative path exists."""
        return self.get_path(relative_path).exists()
    
    def resolve_dataset_path(self, dataset_path: str) -> Path:
        """Resolve dataset path, supports both relative and absolute paths."""
        path = Path(dataset_path)
        
        if path.is_absolute():
            return path
        else:
            full_path = self.get_path(dataset_path)
            if full_path.exists():
                return full_path
            else:
                print(f"Warning: Dataset path does not exist: {full_path}")
                return full_path

path_manager = ProjectPathManager()

def get_project_root() -> Path:
    """Get project root directory."""
    return path_manager.project_root

def get_path(relative_path: Union[str, Path]) -> Path:
    """Get absolute path relative to project root."""
    return path_manager.get_path(relative_path)

def get_dataset_path(dataset_path: Union[str, Path]) -> Path:
    """Get dataset path."""
    return path_manager.get_dataset_path(dataset_path)

def get_config_path(config_file: str = "config/datasets.yaml") -> Path:
    """Get config file path."""
    return path_manager.get_config_path(config_file)

def get_results_path(results_dir: str = "results") -> Path:
    """Get results directory path."""
    return path_manager.get_results_path(results_dir)

def get_cache_path(results_dir: Union[str, Path]) -> Path:
    """Get cache directory path."""
    return path_manager.get_cache_path(results_dir)

def get_irm_cacahe_path(results_dir: Union[str, Path]) -> Path:
    """Get IRM cache directory path."""
    return path_manager.get_irm_cache_path(results_dir)

def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure path exists."""
    return path_manager.ensure_path_exists(path)

def path_exists(relative_path: Union[str, Path]) -> bool:
    """Check if relative path exists."""
    return path_manager.path_exists(relative_path)

def resolve_dataset_path(dataset_path: str | List[str]) -> List[Path]:
    """Resolve dataset path."""
    if isinstance(dataset_path, list):
        return [path_manager.resolve_dataset_path(path) for path in dataset_path]
    else:
        return [path_manager.resolve_dataset_path(dataset_path)]