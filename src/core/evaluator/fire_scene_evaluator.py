import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util
from loguru import logger
from ..base import BaseDataset, BaseEvaluator, BaseModelConfig
from ..model_client import ModelClientFactory
from ...utils.config import ConfigManager
from ...utils.path_manager import get_project_root, get_irm_cacahe_path
import numpy as np
import sys
import jsonlines
sys.path.append(str(get_project_root()))


class _InMemoryCacheWriter:
    """Lightweight cache writer so we can reuse OpenAIModelClient.generate_batch."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def write(self, payload: Dict[str, Any]) -> None:
        self.records.append(payload)


@dataclass
class _FirePrincipleConfig:
    prompt_template: str
    judge_model: str
    prompt_name: str
    prompt_path: Path
    urls: str | list
    api_type: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120
    per_url_max_workers: int = 32
    repeat_num: int = 1
    system_prompt: Optional[str] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = field(default_factory=dict)


class SampleTypeDetector:
    """Detects sample type (principle or rule) based on sample content."""

    @staticmethod
    def determine_sample_type(sample: Dict[str, Any]) -> str:
        """Determine if sample is principle or rule type."""
        if "准则" in sample or "principle" in sample:
            return "principle"
        elif "data_source" in sample:
            return "rule"
        elif "type" in sample:
            return sample["type"]
        else:
            return "unknown"


class RuleEvaluator:
    """Handles rule-based evaluation for different FIRE scene types."""

    @staticmethod
    def extract_answer_from_response(data_source: str, response: Any) -> Any:
        solution_str = response

        if '''```json''' in solution_str:
            match = re.search(r'```json(.*?)(?=```|$)', solution_str, re.DOTALL)
            if match:
                extracted_json = match.group(1).strip()
                solution_str = extracted_json
            else:
                logger.warning(f"==========> {data_source} : no json in response! ")
                return solution_str

        return solution_str

    @staticmethod
    def judge_subtask_router(data_source: str, prediction: Any, ground_truth: Any) -> tuple[bool, str]:
        if "risk" in data_source:
            return RuleEvaluator._judge_risk(prediction, ground_truth), "风控"
        elif "dianxiao" in data_source:
            return RuleEvaluator._judge_dianxiao(prediction, ground_truth), "电销"
        elif "cuishou" in data_source:
            return RuleEvaluator._judge_cuishou(prediction, ground_truth), "催收"
        elif "企业微信" in data_source:
            return RuleEvaluator._judge_qiwei(prediction, ground_truth), "企微"
        elif "金融内容安全拦截" in data_source:
            return RuleEvaluator._judge_content_safe(prediction, ground_truth), "金融内容安全"
        elif "客户对话状态判断_1" in data_source:
            return RuleEvaluator._judge_dialogue_state_classification_1(prediction, ground_truth), "对话状态分类1"
        elif "客户对话状态判断_0" in data_source:
            return RuleEvaluator._judge_dialogue_state_classification(prediction, ground_truth), "对话状态分类0"
        elif "客户反馈归因分析" in data_source:
            return RuleEvaluator._judge_feedback_attribution(prediction, ground_truth), "客户反馈归因"
        elif "客户风险行为预测" in data_source:
            return RuleEvaluator._judge_risk_behavior_prediction(prediction, ground_truth), "风险行为预测"
        elif "客户投诉类型判断_生成" in data_source:
            return RuleEvaluator._judge_complaint_type_classification_gen(prediction, ground_truth), "投诉类型分类生成"
        elif "客户投诉类型判断_抽取" in data_source:
            return RuleEvaluator._judge_complaint_type_classification_extra(prediction, ground_truth), "投诉类型分类抽取"
        elif "推送内容合规" in data_source:
            return RuleEvaluator._judge_push_content_compliance_qc(prediction, ground_truth), "推送内容合规"
        elif "增信话术推荐" in data_source or ("状态判断" in data_source and "客户对话状态判断" not in data_source):
            return RuleEvaluator._judge_credit_talk_recommendation(prediction, ground_truth), "增信话术推荐"
        else:
            raise ValueError("Unknown data source type, no reward category found!")

    @staticmethod
    def _judge_risk(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Risk control: pred error! {e}")
            return False
        ground_truth = float(ground_truth)

        try:
            if prediction["结论"] == "批准放款" and float(ground_truth) == 0:
                correct = True
            elif prediction["结论"] == "拒绝放款" and float(ground_truth) == 1:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Risk control: judge error! {e}")
            return False

    @staticmethod
    def _judge_dianxiao(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Telesales: pred error! {e}")
            return False

        try:
            if prediction["用户情绪状态"] == "用户骂人" and ground_truth == "用户骂人":
                correct = True
            elif prediction["用户情绪状态"] == "用户拒绝客服打电话" and ground_truth == "用户拒绝客服打电话":
                correct = True
            elif prediction["用户情绪状态"] == "其他" and ground_truth == "其他":
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Telesales: judge error! {e}")
            return False

    @staticmethod
    def _judge_qiwei(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"WeCom: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)["router"]
        except Exception as e:
            logger.warning(f"WeCom: gt error! {e}")
            return False

        try:
            if prediction["router"] == ground_truth:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"WeCom: judge error! {e}")
            return False

    @staticmethod
    def _judge_cuishou(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Collections: pred error! {e}")
            return False
        try:
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Collections: gt error! {e}")
            return False

        try:
            if prediction["坐席是否违规"] == ground_truth["坐席是否违规"] and prediction["违规内容"] == ground_truth["违规内容"]:
                correct = True
            elif prediction["坐席是否违规"] == ground_truth["坐席是否违规"]:
                model_tag = set(RuleEvaluator._extract_labels(prediction["违规内容"]))
                answer_tag = set(RuleEvaluator._extract_labels(ground_truth["违规内容"]))
                if model_tag == answer_tag:
                    correct = True
                else:
                    correct = False
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Collections: judge error! {e}")
            return False

    @staticmethod
    def _extract_labels(text):
        """
        Extract all labels before colons from violation content.
        Example: '1、不当话术：威胁联系家人；2、虚构场景：谎称已取消分期资格。'
        Returns: ['不当话术', '虚构场景']
        """
        return re.findall(r'\d+[、\.]?\s*([^：:]+)：', text)

    @staticmethod
    def _judge_complaint_type_classification_extra(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Complaint type classification extra: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Complaint type classification extra: gt error! {e}, {ground_truth}")
            return False

        try:
            if "投诉类型" in ground_truth:
                if prediction["投诉类型"] == ground_truth["投诉类型"]:
                    correct = True
                else:
                    correct = False
            elif "一级分类" in ground_truth:
                if prediction["投诉类型"] == ground_truth["一级分类"]:
                    correct = True
                else:
                    correct = False
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Complaint type classification extra: judge error! {e}")
            return False

    @staticmethod
    def _judge_complaint_type_classification_gen(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Complaint type classification gen: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Complaint type classification gen: gt error! {e}")
            return False

        try:
            if prediction["一级分类"] == ground_truth["一级分类"]:
                if "二级分类" in prediction and "二级分类" in ground_truth and prediction["二级分类"] == ground_truth["二级分类"]:
                    correct = True
                elif "二级分类" not in prediction and "二级分类" not in ground_truth:
                    correct = True
                elif "二级分类" not in prediction and "二级分类" in ground_truth and ground_truth["二级分类"] == "":
                    correct = True
                else:
                    correct = False
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Complaint type classification gen: judge error! {e}")
            return False

    @staticmethod
    def _judge_content_safe(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Content safe: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "").replace("，", ",").replace("：",":").replace("/n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Content safe: gt error! {e}")
            return False

        try:
            if prediction["文件类型"] == ground_truth["文件类型"] and prediction["是否公司备案数据"] == ground_truth["是否公司备案数据"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Content safe: judge error! {e}")
            return False

    @staticmethod
    def _judge_credit_talk_recommendation(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Credit talk recommendation: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Credit talk recommendation: gt error! {e}")
            return False

        try:
            if prediction["router"] == ground_truth["router"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Credit talk recommendation: judge error! {e}")
            return False

    @staticmethod
    def _judge_dialogue_state_classification_1(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Dialogue state classification 1: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Dialogue state classification 1: gt error! {e}")
            return False

        try:
            if prediction["类别id"] == ground_truth["类别id"] and prediction["类别内容"] == ground_truth["类别内容"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Dialogue state classification 1: judge error! {e}")
            return False

    @staticmethod
    def _judge_dialogue_state_classification(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Dialogue state classification 0: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Dialogue state classification 0: gt error! {e}")
            return False

        try:
            if prediction["当前所处议价环节"] == ground_truth["当前所处议价环节"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Dialogue state classification 0: judge error! {e}")
            return False

    @staticmethod
    def _judge_feedback_attribution(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Feedback attribution: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Feedback attribution: gt error! {e}")
            return False

        try:
            if prediction["所属业务线"] == ground_truth["所属业务线"] and prediction["通话类型"] == ground_truth["通话类型"] and prediction["是否纳入信贷业务投诉分析"] == ground_truth["是否纳入信贷业务投诉分析"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Feedback attribution: judge error! {e}")
            return False

    @staticmethod
    def _judge_push_content_compliance_qc(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Push content compliance qc: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Push content compliance qc: gt error! {e}")
            return False

        try:
            for key in prediction.keys():
                if key in ground_truth:
                    if prediction[key]["是否合规"] == ground_truth[key]["是否合规"]:
                        correct = True
                    else:
                        correct = False
                        break
                else:
                    correct = False
                    break
            return correct
        except Exception as e:
            logger.warning(f"Push content compliance qc: judge error! {e}")
            return False

    @staticmethod
    def _judge_risk_behavior_prediction(prediction: Any, ground_truth: Any) -> bool:
        try:
            prediction = json.loads(prediction.replace("\n",""))
        except Exception as e:
            logger.warning(f"Risk behavior prediction: pred error! {e}")
            return False
        try:
            ground_truth = ground_truth.replace("```json","").replace("```","").replace("\n", "")
            ground_truth = json.loads(ground_truth)
        except Exception as e:
            logger.warning(f"Risk behavior prediction: gt error! {e}")
            return False

        try:
            if prediction["standard_category"] == ground_truth["standard_category"] and prediction["standard_id"] == ground_truth["standard_id"]:
                correct = True
            else:
                correct = False
            return correct
        except Exception as e:
            logger.warning(f"Risk behavior prediction: judge error! {e}")
            return False


class SceneAggregator:
    """Handles scene aggregation using task2scene mappings."""


    def aggregate_by_scene(
        self,
        scores: List[Optional[float]],
        data_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate scores by scene (primary and secondary levels) using task2scene mapping."""
        primary_scene_scores: Dict[str, List[float]] = defaultdict(list)
        secondary_scene_scores: Dict[str, List[float]] = defaultdict(list)

        principle_scene_scores = []
        rule_scene_scores = []

        error_eval_scene_counter = defaultdict(int)
        scene2top = {}

        for score, sample in zip(scores, data_samples):
            if score is None:
                error_eval_scene_counter[f"{sample['top_name']}"] += 1
                continue



            secondary_scene = f"{sample['task_name']}"
            primary_scene = f"{sample['top_name']}"

            primary_scene_scores[primary_scene].append(score)
            secondary_scene_scores[secondary_scene].append(score)
            scene2top[secondary_scene] = primary_scene

            if "准则" in sample or "principle" in sample:
                principle_scene_scores.append(score)
            else:
                rule_scene_scores.append(score)

        primary_averages = {
            scene: (sum(scores_list) / len(scores_list) if scores_list else 0.0)
            for scene, scores_list in primary_scene_scores.items()
        }

        secondary_averages = defaultdict(dict)
        for scene, scores_list in secondary_scene_scores.items():
            secondary_averages[scene2top[scene]][scene] = (sum(scores_list) / len(scores_list) if scores_list else 0.0)

        principle_scene_avergae = (sum(principle_scene_scores) / len(principle_scene_scores) if principle_scene_scores else 0.0)
        rule_scene_avergae = (sum(rule_scene_scores) / len(rule_scene_scores) if rule_scene_scores else 0.0)
        
        def parse_task_id(task_id_str):
            """Parse task_id string and return sortable tuple."""
            try:
                parts = task_id_str.replace('-', '.').split('.')
                return tuple(int(x) for x in parts if x.isdigit())
            except (ValueError, AttributeError):
                return (task_id_str,)

        def sort_key(scene):
            """Sort key function: extract task_id and parse to sortable format."""
            task_id = scene.split('_')[0]
            return parse_task_id(task_id)

        secondary_averages = dict(sorted(secondary_averages.items(), key=lambda x: x[0]))
        for top_name, scenes in secondary_averages.items():
            secondary_averages[top_name] = dict(sorted(scenes.items(), key=lambda x: sort_key(x[0])))

        return {
            "primary_scenes": primary_averages,
            "secondary_scenes": secondary_averages,
            "principle_scene_avergae": principle_scene_avergae,
            "rule_scene_avergae": rule_scene_avergae,
            "error_secondary_scenes_counter": error_eval_scene_counter,
        }


class FireSceneEvaluator(BaseEvaluator):
    """Evaluator that scores FIRE scene responses with rubric-aware judge model or rules."""

    def __init__(self, dataset_config: Optional[BaseDataset] = None, **kwargs: Any):
        super().__init__(dataset_config, **kwargs)
        if dataset_config is None:
            raise ValueError("FireSceneEvaluator requires a dataset_config instance.")

        # Use provided config_manager or create a new one
        self._config_manager = kwargs.get('config_manager') or ConfigManager()
        self._raw_config = self._resolve_raw_config(dataset_config)
        self._settings = self._build_settings(self._raw_config)
        self._model_client_factory = ModelClientFactory()

        # Initialize helper classes
        self._type_detector = SampleTypeDetector()
        self._rule_evaluator = RuleEvaluator()
        self._scene_aggregator = SceneAggregator()
        self.irm_cache_dir: Optional[Path] = None
    # -------- Prompt & config helpers -------------------------------------------------
    def _resolve_raw_config(self, dataset_config: BaseDataset) -> Dict[str, Any]:
        datasets_cfg = self._config_manager.datasets_config.get("datasets", {})
        key = dataset_config.name

        if key in datasets_cfg:
            return datasets_cfg[key]

        for cfg_key, cfg_val in datasets_cfg.items():
            if cfg_val.get("name") == key:
                return cfg_val

        raise KeyError(f"Dataset config for '{dataset_config.name}' not found in datasets.yaml")

    def _build_settings(self, raw_config: Dict[str, Any]) -> _FirePrincipleConfig:
        prompt_path = raw_config.get("prompt_path")
        prompt_name = raw_config.get("prompt_name")
        if not prompt_path or not prompt_name:
            raise ValueError("FIRE principle dataset must configure both prompt_path and prompt_name.")

        prompt_template = self._load_prompt_template(prompt_path, prompt_name)
        judge_model = raw_config.get("judge_model", raw_config.get("judge_model", raw_config.get("llm_model")))
        if not judge_model:
            raise ValueError("FIRE principle dataset must configure judge_model.")


        settings = _FirePrincipleConfig(
            prompt_template=prompt_template,
            judge_model=judge_model,
            prompt_name=prompt_name,
            prompt_path=get_project_root() / prompt_path,
            urls=raw_config.get("judge_model_urls", []),
            api_type=raw_config.get("judge_model_api_type", ""),
            api_key=raw_config.get("judge_model_api_key", ""),
            temperature=float(raw_config.get("judge_temperature", 0.6)),
            max_tokens=int(raw_config.get("judge_max_tokens", 2048)),
            timeout=int(raw_config.get("judge_timeout", 300)),
            repeat_num=int(raw_config.get("judge_repeat_num", 1)),
            per_url_max_workers=int(raw_config.get("judge_per_url_max_workers", 32)),
            system_prompt=raw_config.get("judge_system_prompt", None),
            top_p=raw_config.get("judge_top_p", None),
            top_k=raw_config.get("judge_top_k", None),
            extra_body=raw_config.get("judge_extra_body", {}),
        )
        return settings

    def _load_prompt_template(self, relative_path: str, prompt_name: str) -> str:
        prompt_file = get_project_root() / relative_path
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        module_name = f"fire_principle_prompt_{hash(prompt_file)}"
        spec = importlib.util.spec_from_file_location(module_name, prompt_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {prompt_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        template = getattr(module, prompt_name, None)
        if template is None:
            raise AttributeError(f"Prompt '{prompt_name}' not found in {prompt_file}")

        return template

    def _build_model_config(self) -> BaseModelConfig:
        model_key = self._settings.judge_model
        

        config = BaseModelConfig(
            name=f"fire_principle_judge_{model_key}",
            urls=self._settings.urls,
            per_url_max_workers=self._settings.per_url_max_workers,
            api_key=self._settings.api_key,
            api_type=self._settings.api_type,
            model=model_key,
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
            timeout=self._settings.timeout,
            extra_body=self._settings.extra_body,
            system_prompt=self._settings.system_prompt,
            streaming=False,
            use_chat=True,
        )
    
        return config

    def _process_model_response(self, response: str) -> str:
        """
        Process model response, extract content after <think></think> tags.
        
        Args:
            response: Raw model response
            
        Returns:
            Processed response content
        """
        think_start = response.find('<think>')
        think_end = response.find('</think>')
        end_len = len('</think>')

        if think_start == -1 and think_end == -1:  
            think_start = response.find('<seed:think>')
            think_end = response.find('</seed:think>')
            end_len = len('</seed:think>')
        
        if think_start != -1:
            if think_end == -1:
                result = ""
            else:
                result = response[think_end + end_len:].lstrip()
        else:
            if think_end != -1:
                result = response[think_end + end_len:].lstrip()
            result = response
        logger.debug(f"Processed response: {result}")
        return result

    # -------- BaseEvaluator overrides --------------------------------------------------
    def extract_format_prompt(self, dataset_config: BaseDataset, sample: Dict[str, Any]) -> str:
        instruction = (
            sample.get("filled_prompt")
            or sample.get("prompt")
            or sample.get("instruction")
            or sample.get("question")
        )
        if not instruction:
            raise ValueError("Sample does not contain a usable prompt field (filled_prompt/question).")
        if isinstance(instruction, list) or isinstance(instruction, np.ndarray):
            instruction = instruction[0].get("content", "")
        return str(instruction)

    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        sample_type = self._type_detector.determine_sample_type(sample)
        if sample_type == "principle":
            ground_truth = sample.get("human_score")
            return str(ground_truth)
        elif sample_type == "rule":
            ground_truth = sample.get("reward_model").get("ground_truth")
            return str(ground_truth)
        else:
            raise ValueError("Unknown sample type")

    # -------- Evaluation loop ----------------------------------------------------------
    async def evaluate_async(
        self,
        predictions: List[str],
        ground_truths: List[str],
        data_samples: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if data_samples is None:
            raise ValueError("FireSceneEvaluator requires original samples for aggregation.")

        if len(predictions) != len(data_samples):
            raise ValueError("Predictions and data samples must align for FIRE scene evaluation.")

        total = len(predictions)
        scores = [None] * total
        principle_samples = []
        principle_indices = []

        repeat_num = self._settings.repeat_num

        for i, (sample, pred, gt) in enumerate(zip(data_samples, predictions, ground_truths)):
            if pred == "" or pred is None:
                continue
            sample_type = self._type_detector.determine_sample_type(sample)
            if sample_type == "principle":
                principle_samples.append((sample, pred, gt))
                principle_indices.append(i)
            elif sample_type == "rule":
                data_source = sample.get('data_source', 'unknown')
                answer = self._rule_evaluator.extract_answer_from_response(data_source, pred)
                is_correct, subtask = self._rule_evaluator.judge_subtask_router(data_source, answer, gt)
                score = 100.0 if is_correct else 0.0
                scores[i] = score
            else:
                logger.warning(f"Unknown sample type for sample {i}, treating as invalid")
                scores[i] = None

        if principle_samples:
            principle_prompts = [
                self._format_judge_prompt(sample, pred, gt)
                for sample, pred, gt in principle_samples
            ]

            ori_prompts_len = len(principle_prompts)
            logger.warning(f"now start using judge repeat_num :{repeat_num}!")
            principle_prompts = principle_prompts * repeat_num

            if principle_prompts:
                model_config = self._build_model_config()

                cache_dict_list = []
                cache_file = None
                if self.irm_cache_dir is not None:
                    cache_file = self.irm_cache_dir / f"{model_config.name}.json"
                    if cache_file.exists():
                        logger.info(f"Principle cache file exists: {cache_file}, will load from cache")
                        with jsonlines.open(cache_file) as reader:
                            cache_dict_list = list(reader)
                    else:
                        logger.info(f"Principle cache file does not exist: {cache_file}, will generate cache")

                principle_prompts_candi = []
                principle_prompts_cached = []
                judge_outputs_cached = []
                candi_indices = []

                for i, prompt in enumerate(principle_prompts):
                    cached_response = self._get_principle_response_from_cache(prompt, cache_dict_list)
                    if cached_response:
                        principle_prompts_cached.append(prompt)
                        judge_outputs_cached.append(cached_response)
                    else:
                        principle_prompts_candi.append(prompt)
                        candi_indices.append(i)

                logger.info(f"Principle processing: {len(principle_prompts_candi)} new prompts, {len(principle_prompts_cached)} cached prompts")

                model_client = self._model_client_factory.create_client(model_config)

                judge_outputs_candi = []
                if principle_prompts_candi:
                    if self.irm_cache_dir is not None and cache_file:
                        cache_writer = jsonlines.Writer(open(cache_file, mode='a', encoding='utf-8'))
                    else:
                        cache_writer = _InMemoryCacheWriter()

                    if not model_client.validate_config(model_config):
                        raise ValueError("Judge model configuration validation failed.")

                    async with model_client:
                        judge_outputs_candi = await model_client.generate_batch(
                            principle_prompts_candi,
                            cache_writer=cache_writer,
                            temperature=self._settings.temperature,
                            max_tokens=self._settings.max_tokens,
                            top_p=self._settings.top_p,
                            top_k=self._settings.top_k,
                        )

                judge_outputs: List[str] = [""] * len(principle_prompts)
                cached_idx = 0
                for i, prompt in enumerate(principle_prompts):
                    if i not in candi_indices:
                        judge_outputs[i] = judge_outputs_cached[cached_idx]
                        cached_idx += 1

                for candi_idx, output in zip(candi_indices, judge_outputs_candi):
                    judge_outputs[candi_idx] = output

                principle_scores, parsing_failures = self._parse_scores(judge_outputs)
                if parsing_failures:
                    logger.warning(f"Failed to parse scores for {parsing_failures} / {len(judge_outputs)} principle items.")

                chunks = [principle_scores[i*ori_prompts_len : (i+1)*ori_prompts_len] for i in range(repeat_num)]
                averages = []
                for group in zip(*chunks):
                    valid_nums = [x for x in group if x is not None]
                    
                    if valid_nums:
                        averages.append(sum(valid_nums) / len(valid_nums))
                    else:
                        averages.append(None)
                principle_scores = averages
                assert len(principle_indices) == len(principle_scores), ""

                for idx_pos, sample_idx in enumerate(principle_indices):
                    score = principle_scores[idx_pos]
                    if score is not None:
                        normalized_score = ((score - 1) / 4.0) * 100.0
                        scores[sample_idx] = normalized_score
                    else:
                        scores[sample_idx] = None

        valid_scores = [score for score in scores if score is not None]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        score_distribution = Counter(int(score) for score in valid_scores if score is not None)

        scene_averages = self._scene_aggregator.aggregate_by_scene(scores, data_samples)

        result = {
            "average_score": average_score,
            "total": total,
            "valid_scores": len(valid_scores),
            "invalid_scores": total - len(valid_scores),
            "scene_averages": scene_averages,
            "score_distribution": dict(score_distribution),
            "is_correct_list": scores,
        }

        return result

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return asyncio.run(self.evaluate_async(predictions, ground_truths, **kwargs))

    # -------- Internal helpers ---------------------------------------------------------
    def _format_judge_prompt(self, sample: Dict[str, Any], prediction: str, ground_truth: str) -> str:
        instruction = (
            sample.get("filled_prompt")
            or sample.get("prompt")
            or sample.get("instruction")
            or sample.get("question")
            or ""
        )
        reference_answer = sample.get("参考答案") or sample.get("ref_answer") or ground_truth or ""
        if reference_answer == "" or reference_answer is None or reference_answer == "None":
            reference_answer = ""
        principle = sample.get("准则") or sample.get("principle") or ""
        response = prediction or ""

        prompt = self._settings.prompt_template.format(
            instruction=str(instruction).strip(),
            ref_answer=str(reference_answer).strip(),
            principle=str(principle).strip(),
            response=str(response).strip(),
        )
        return prompt

    def _parse_scores(self, judge_outputs: List[str]) -> tuple[List[Optional[float]], int]:
        scores: List[Optional[float]] = []
        failures = 0

        for output in judge_outputs:
            output = self._process_model_response(output)
            score = self._extract_score(output)
            if score is None:
                failures += 1
            scores.append(score)

        return scores, failures

    def _extract_score(self, response: str) -> Optional[float]:
        if not response:
            return None

        response = response.strip()

        json_block = self._extract_json_block(response)

        try:
            if json_block is None:
                return None
            parsed = json.loads(json_block)
            score_val = parsed.get("score")
            if isinstance(score_val, (int, float)):
                return float(score_val)
            if isinstance(score_val, str) and score_val.strip():
                return float(score_val.strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        match = re.search(r'"score"\s*:\s*([-+]?\d+(?:\.\d+)?)', response)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

        standalone_match = re.search(r'\b([1-5](?:\.\d+)?)\b', response)
        if standalone_match:
            try:
                return float(standalone_match.group(1))
            except ValueError:
                return None

        return None

    def _get_principle_response_from_cache(self, prompt: str, cache_dict_list: List[Dict[str, Any]]) -> Optional[str]:
        """Get principle judge response from cache"""
        if not cache_dict_list:
            return None

        for i, cache_dict in enumerate(cache_dict_list):
            if cache_dict["prompt"] == prompt:
                matched_item = cache_dict_list.pop(i)
                return matched_item["model_response"]
        return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return None


