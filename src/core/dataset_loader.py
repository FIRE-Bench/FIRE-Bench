"""
Simplified JSON dataset loader for multi-format data processing
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger


from .base import BaseDatasetLoader, BaseDataset
from ..utils.path_manager import resolve_dataset_path, get_project_root


class DatasetLoader(BaseDatasetLoader):
    """Simplified loader for JSON format datasets"""
    
    def load(self, dataset_config: BaseDataset) -> List[Dict[str, Any]]:
        """Load JSON dataset with support for various formats"""
        try:
            # Use path manager to resolve dataset paths
            dataset_paths = resolve_dataset_path(dataset_config.path)
            logger.info(f"Loading dataset from: {dataset_paths}")
            total_data = []
            for dataset_path in dataset_paths:
                if dataset_path.is_file():
                    # Single file - JSON or JSONL
                    data = self._load_single_file(dataset_path)
                elif dataset_path.is_dir():
                    # Directory with JSON/JSONL files or Excel files
                    data = self._load_directory(dataset_path)
                else:
                    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
                total_data.extend(data)
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config.name}: {str(e)}")
            raise
        return total_data
    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single file (JSON, JSONL, or Excel...)"""

        if file_path.suffix.lower() == '.jsonl':
            return self._load_jsonl(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._load_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return self._load_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._load_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return self._load_parquet(file_path)
        else:
            # Try to load as JSON by default
            return self._load_json(file_path)

    
    def _load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Load all compatible files from directory and subdirectories, with subtask support for FinanceIQ, Fineval and Finova"""
        all_data = []
        file_patterns = ['*.json', '*.jsonl', '*.xlsx', '*.xls', '*.csv', '*.parquet']
        for pattern in file_patterns:
            # Use rglob to recursively search subdirectories
            for file_path in dir_path.rglob(pattern):
                try:
                    file_data = self._load_single_file(file_path)
                    all_data.extend(file_data)
                    logger.info(f"Loaded {len(file_data)} samples from {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")
                    continue
        if not all_data:
            raise ValueError(f"No valid data files found in {dir_path}")
        return all_data
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        """Extract subject name from CSV filename for Fineval"""
        # Remove _val.csv or _test.csv suffix
        subject = filename.replace('_val.csv', '').replace('_merged.csv', '').replace('_dev.csv', '')
        return subject

    def _detect_encoding(self, file_path: Path) -> str:
        """Auto-detect file encoding"""

        # Fallback: try common encodings
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)
                logger.info(f"Using encoding: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'
        
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file"""
        
        encoding = self._detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load JSON file {file_path}: {str(e)}")
            data = pd.read_json(file_path)
        
        if isinstance(data, str):
            data = json.loads(data)
        
        # Normalize data format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Handle different JSON structures
            if 'data' in data:
                return data['data'] if isinstance(data['data'], list) else [data['data']]
            elif 'examples' in data:
                return data['examples'] if isinstance(data['examples'], list) else [data['examples']]
            elif 'items' in data:
                return data['items'] if isinstance(data['items'], list) else [data['items']]
            else:
                return [data]
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {str(e)}")
                        continue
        return data
    
    def _load_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Excel file using pandas"""
        try:
            df = pd.read_excel(file_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load Excel file {file_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file using pandas"""
        try:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {str(e)}")
            raise

    def _load_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Parquet file using pandas"""
        try:
            df = pd.read_parquet(file_path)
            result = df.to_dict('records')
            return result
        except Exception as e:
            logger.error(f"Failed to load Parquet file {file_path}: {str(e)}")
            raise
    
    
    def validate(self, dataset_config: BaseDataset) -> bool:
        """Validate JSON dataset"""
        # Use path manager to resolve dataset paths
        dataset_paths = [path for path in resolve_dataset_path(dataset_config.path)]
        return all(self._validate_file(path) for path in dataset_paths)
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate a single file"""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
            elif file_path.suffix.lower() == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Check first line
                    first_line = f.readline().strip()
                    if first_line:
                        json.loads(first_line)
                return True
            elif file_path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
                # For Excel/CSV, just check if file is readable
                return file_path.stat().st_size > 0
            elif file_path.suffix.lower() == '.parquet':
                # For Parquet, just check if file is readable
                return file_path.stat().st_size > 0
            else:
                # Try as JSON by default
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
                
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False