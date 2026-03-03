"""
Dataset evaluation pipeline - simplified for data processing only
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import shutil
import jsonlines
import os
import random
from .base import BaseModelConfig, BaseDataset, EvaluationResult
from .model_client import ModelClientFactory, OpenAIModelClient
from .dataset_loader import DatasetLoader
from .evaluator import evaluator_manager
from ..utils.path_manager import get_results_path, get_path, get_cache_path, get_irm_cacahe_path
import traceback
from ..utils.config import ConfigManager
import re

def process_model_response(response: str) -> str:
    """
    Process model response, extract content after <think></think> tags

    Args:
    response (str): The model's raw response

    Returns:
    str: Processed response content
    """
    # Check if there are <think> tags
    think_start = response.find('<think>')
    think_end = response.find('</think>')
    end_len = len('</think>')

    if think_start == -1 and think_end == -1:  
        think_start = response.find('<seed:think>')
        think_end = response.find('</seed:think>')
        end_len = len('</seed:think>')
    
    if think_start == -1 and think_end == -1:  
        think_start = response.find('<thinking>')
        think_end = response.find('</thinking>')
        end_len = len('</thinking>')
    
    if think_start != -1: 
        if think_end == -1: 
            result = ""
        else:  
            result = response[think_end + end_len:].lstrip()
    else: 
        if think_end != -1: 
            result = response[think_end + end_len:].lstrip()
        else:
            result = response
    # adapt xuanyuan-fin-x1、dianjing-r1
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', result.strip(), re.DOTALL)
    if match:
        return match.group(1)
    logger.debug(f"Processed response: {result}")
    return result

class EvaluationPipeline:
    """Dataset evaluation pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, results_dir: str = "results"):
        self.model_client_factory = ModelClientFactory()
        self.dataset_loader = DatasetLoader()
        self.evaluator_manager = evaluator_manager
        self.config_path = config_path
        self.results_dir = get_results_path(results_dir)
        self.cache_dir = None
        self.timestamp = None
        self.results_folder = None
    
    async def run_evaluation(self, 
                           config_manager: ConfigManager,
                           model_config: BaseModelConfig,
                           dataset_names: List[str],
                           max_samples: Optional[int] = None,
                           results_dir: str = "results",
                           resume_folder: Optional[str] = None) -> List[EvaluationResult]:
        """Run evaluation on datasets"""
        results = []

        dataset_configs = self._load_dataset_configs(config_manager, dataset_names)

        if resume_folder:
            self.results_folder = get_path(resume_folder)
            self.timestamp = self.results_folder.name.split('_')[-1]
            self.results_dir = self.results_folder.parent
            logger.warning(f"Resume directory is: {self.results_dir}, will save results to {self.results_dir}, original result dir is ignored!")
        else:
            # Update results directory if provided
            self.timestamp = datetime.now().strftime("%m%d%H%M")
            self.results_dir = get_results_path(results_dir)
            self.results_folder = self.results_dir / f"{model_config.name}_{'-'.join(dataset_names)}_{self.timestamp}"
            logger.info(f"Results directory updated to: {self.results_dir}")
        self.results_folder.mkdir(parents=True, exist_ok=True)
        resume_datasets_yaml = self.results_folder / "datasets.yaml"
        if not os.path.exists(resume_datasets_yaml):
            shutil.copy(config_manager.config_file, resume_datasets_yaml)
        logger.info(f"Results will be saved to: {self.results_folder}")
        self.cache_dir = get_cache_path(self.results_folder)
        logger.info(f"Cache will be saved to: {self.cache_dir}")
        
        model_client = self.model_client_factory.create_client(model_config)
        
        if not model_client.validate_config(model_config):
            raise ValueError("Invalid model configuration")
        
        async with model_client:
            for dataset_config in dataset_configs:
                logger.info(f"Evaluating on dataset: {dataset_config.name}")
                
                try:
                    result = await self._evaluate_single_dataset(
                        model_client, model_config, dataset_config, max_samples, config_manager
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {dataset_config.name}: {str(e)}")
                    traceback.print_exc()
                    continue
                logger.info(f"========================================================")
        
        self._save_results(results, model_config.name, config_manager)
        
        return results
    
    async def _evaluate_single_dataset(self,
                                     model_client: OpenAIModelClient,
                                     model_config: BaseModelConfig,
                                     dataset_config: BaseDataset,
                                     max_samples: Optional[int],
                                     config_manager: ConfigManager) -> EvaluationResult:
        """Evaluate on a single dataset"""

        cache_file = self.cache_dir / f"{model_config.name}_{dataset_config.name}.json"
        if cache_file.exists():
            logger.info(f"Cache file exists: {cache_file}, will load from cache")
            with jsonlines.open(cache_file) as reader:
                cache_dict_list = list(reader)
        else:
            cache_dict_list = []
            logger.info(f"Cache file does not exist: {cache_file}, will generate cache")

        evaluator = self.evaluator_manager.build(dataset_config.evaluator, dataset_config=dataset_config, config_manager=config_manager)
        if hasattr(evaluator, 'evaluator'):
            evaluator.evaluator.irm_cache_dir = get_irm_cacahe_path(self.results_folder)
        dataset = self.dataset_loader.load(dataset_config)
        print(f"Dataset {dataset_config.name} has {len(dataset)} samples")
        if dataset_config.shuffle:
            random.shuffle(dataset)
        repeat_num = getattr(dataset_config, 'repeat_num', 1)
        if repeat_num > 1:
            original_size = len(dataset)
            dataset = dataset * repeat_num
            logger.info(f"Dataset {dataset_config.name} repeated {repeat_num} times: {original_size} -> {len(dataset)} samples")
        else:
            logger.info(f"Dataset {dataset_config.name} using original data without repetition")
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        logger.info(f"Loaded {len(dataset)} samples from {dataset_config.name}")
        logger.info(f"Using Streaming : {model_config.streaming}")
        logger.info(f"Is Chat model: {model_config.use_chat}")
        

        prompts_candi, prompts_cached = [], []
        ground_truths_candi, ground_truths_cached = [], []
        valid_samples_candi, valid_samples_cached = [], [] 
        predictions_cached = []
        for i, sample in enumerate(dataset):
            try:
                prompt = evaluator.extract_format_prompt(dataset_config, sample)
                ground_truth = evaluator.extract_ground_truth(sample)

                model_response_cached = self._get_model_response_from_cache(prompt, cache_dict_list)
                if model_response_cached:
                    prompts_cached.append(prompt)
                    ground_truths_cached.append(ground_truth)
                    valid_samples_cached.append(sample)
                    predictions_cached.append(model_response_cached)
                else:
                    prompts_candi.append(prompt)
                    ground_truths_candi.append(ground_truth)
                    valid_samples_candi.append(sample)
            except Exception as e:
                logger.warning(f"Error processing sample: {str(e)}")
                continue
        
        if not prompts_candi and not prompts_cached:
            raise ValueError(f"No valid samples found in dataset {dataset_config.name}")
        
        logger.info(f"Processing {len(prompts_candi)} valid samples, {len(prompts_cached)} cached samples")
        
        try:
            cache_writer = jsonlines.Writer(open(cache_file, mode='a', encoding='utf-8'))
            predictions_candi = await model_client.generate_batch(prompts_candi, cache_writer=cache_writer)
        except Exception as e:
            logger.error(f"Error in generate_batch: {str(e)}")
            raise e
        finally:
            pass
            
        
        prompts = prompts_candi + prompts_cached
        predictions = predictions_candi + predictions_cached
        ground_truths = ground_truths_candi + ground_truths_cached
        valid_samples = valid_samples_candi + valid_samples_cached

        filtered_data = [(p, g, s) for p, g, s in zip(predictions, ground_truths, valid_samples)]
        filtered_data = [(process_model_response(p), g, s) for p, g, s in filtered_data]

        logger.info(f"Streaming batch processing completed: {len([r for r in predictions if r.strip()])}/{len(predictions)} successful, but calculate use total samples")

        if filtered_data:
            filtered_predictions, filtered_ground_truths, filtered_valid_samples = map(list, zip(*filtered_data))
            logger.info("Evaluate all samples, including empty predictions")

            if hasattr(evaluator, 'evaluate_async'):
                evaluate_async_func = getattr(evaluator, 'evaluate_async')
                eval_results = await evaluate_async_func(filtered_predictions, filtered_ground_truths, data_samples=filtered_valid_samples)
            else:
                eval_results = evaluator.evaluate(filtered_predictions, filtered_ground_truths, data_samples=filtered_valid_samples)
        else:
            logger.warning("All predictions are empty")
            eval_results = {"accuracy": 0.0, "total": len(predictions), "valid": 0}
        
        metrics = {
            "request_success_rate": len([r for r in predictions if r.strip()]) / len(predictions),
        }
        sample_stats = {}
        
        is_correct_list = eval_results.get("is_correct_list", [None] * len(predictions))
        if eval_results.get("is_correct_list"):
            del eval_results["is_correct_list"]
        
        excluded_keys = {'is_correct_list'}
        for key, value in eval_results.items():
            if key not in excluded_keys:
                metrics[key] = value
        
        result = EvaluationResult(
            dataset_name=dataset_config.name,
            model_name=model_config.name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            sample_stats=sample_stats if sample_stats else None,
            details={
                "filled_prompts": prompts,
                "raw_samples": valid_samples,
                "predictions": predictions,
                "eval_results": eval_results,
                "is_correct_list": is_correct_list
            }
        )
        
        return result
    
    def _get_model_response_from_cache(self, prompt: str, cache_dict_list: List[Dict[str, Any]]) -> Optional[str]:
        """Get model response from cache"""
        if not cache_dict_list:
            return None
        
        for i, cache_dict in enumerate(cache_dict_list):
            if cache_dict["prompt"] == prompt:
                matched_item = cache_dict_list.pop(i)
                return matched_item["model_response"]
        return None


    def _load_dataset_configs(self, config_manager: ConfigManager, dataset_names: List[str]) -> List[BaseDataset]:

        """Loa dataset configurations"""
        
        configs: list[Any] = []
        
        for name in dataset_names:
            config = config_manager.get_dataset_config(name)
            if not config:
                logger.error(f"Dataset configuration not found: {name}")
                continue
                
            configs.append(config)
        
        return configs
    
    def _save_results(self, results: List[EvaluationResult], model_name: str, config_manager: ConfigManager):
        """Save evaluation results"""
        if not results:
            logger.warning("No results to save")
            return
        
        results_folder = self.results_folder
        timestamp = self.timestamp
        
        summary_file = results_folder / "metrics_summary.json"
        results_dict = {
            "model_name": model_name,
            "timestamp": timestamp,
            "total_datasets": len(results),
            "results": [r.dict(exclude={"details"}) for r in results]
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics summary: {str(e)}")
        
        for result in results:
            if result.details and "raw_samples" in result.details and "predictions" in result.details:
                try:
                    dataset_file = results_folder / f"{model_name}_{result.dataset_name}.json"
                    
                    enhanced_samples = []
                    filled_prompts = result.details["filled_prompts"]
                    raw_samples = result.details["raw_samples"]
                    predictions = result.details["predictions"]
                    is_correct_list = result.details["is_correct_list"]
                    
                    for i, (filled_prompt, sample, prediction, is_correct) in enumerate(zip(filled_prompts, raw_samples, predictions, is_correct_list)):
                        enhanced_sample = sample.copy()
                        enhanced_sample["filled_prompt"] = filled_prompt
                        enhanced_sample["model_response"] = prediction
                        enhanced_sample["is_correct"] = is_correct
                        enhanced_samples.append(enhanced_sample)
                    
                    dataset_content = {
                        "dataset_name": result.dataset_name,
                        "model_name": model_name,
                        "timestamp": result.timestamp,
                        "total_samples": len(enhanced_samples),
                        "metrics": result.metrics,
                        "data": enhanced_samples
                    }
                    
                    with open(dataset_file, 'w', encoding='utf-8') as f:
                        json.dump(dataset_content, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Dataset file saved: {dataset_file}")
                
                except Exception as e:

                    logger.error(f"Failed to save dataset file for {result.dataset_name}: {str(e)}")
            else:
                logger.warning(f"No raw data available for dataset: {result.dataset_name}")
        
        logger.info(f"All results saved to folder: {results_folder}")