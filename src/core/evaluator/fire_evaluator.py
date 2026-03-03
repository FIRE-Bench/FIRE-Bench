import re
from typing import Dict, List, Any, Optional
from loguru import logger
from ..base import BaseEvaluator, BaseDataset
from collections import defaultdict
import random


class FIREMCQEvaluator(BaseEvaluator):
    def __init__(self, dataset_config: Optional[BaseDataset] = None, demo_count: int = 0, **kwargs):

        self.dataset_config = dataset_config
        self.demo_count = max(0, min(5, demo_count))

    def extract_format_prompt(self, dataset_config: BaseDataset, sample: Dict[str, Any]) -> str:

        prompt = sample.get('prompt', '')
        question = sample.get('question', '')
        demos = sample.get('demo', [])
        if not question or not prompt:
            raise ValueError("FIRE sample missing question or prompt")

        formatted_prompt = prompt + "请直接给出正确答案的选项，注意：结果只需要返回答案的英文选项，输出格式限制为：\"答：选项\"，如 \"答：ABC\"\n"
        
        if demos and self.demo_count > 0:
            selected_demos = demos[:self.demo_count]
            for demo in selected_demos:
                formatted_prompt += f"{demo}\n"
        
        formatted_prompt += question
        
        return formatted_prompt

    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        
        answer = sample.get('gold')
        if not answer:
            raise ValueError("FIRE sample missing gold answer")

        if isinstance(answer, str):
            return answer.strip().upper()
        raise ValueError("FIRE sample missing valid gold answer")

    def evaluate(self, predictions: List[str], ground_truths: List[str], data_samples: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, float]:

        def extract_choice(response):
            response = str(response).strip().upper()
            # Add preprocessing step: replace all possible separators like "and", spaces, Chinese and English commas, pauses with no separators
            response = response.replace('和', '').replace('与', '').replace(' ', '').replace('、', '').replace(',', '').replace('，', '')
            choices = 'ABCDE'
            
            patterns = [
                (r'答[：:]?([A-E]+)', 1),
                (r'答案(选项)?(是|为)?[：:]? ?([A-E]+)', 3),
                (r'答案(是|为)选项 ?([A-E]+)', 2),
                # Handle answer formats containing Chinese pauses, such as "Answer: A, B, C"
                (r'答案(选项)?(是|为)[：:]? ?([A-E])(?:、([A-E]))*(?=\s*$)', lambda m: ''.join(g for g in m.groups()[2:] if g)),
                (r'^\s*([A-E]+)', 1), # Added: The beginning is a continuous string consisting only of A-E.
            ]
            
            
            matched_choices = []
            for pattern, idx_or_fn in patterns:
                m = re.search(pattern, response, re.M)
                if m:
                    if callable(idx_or_fn):
                        # If it's a function, call the function to process the match result
                        answer = idx_or_fn(m)
                    else:
                        # Otherwise use numeric index to get match group
                        answer = m.group(idx_or_fn)
                    if answer and all(c in choices for c in answer):
                        matched_choices.extend(answer)
            
            if matched_choices:
                unique_choices = sorted(list(set(matched_choices)))
                return ''.join(unique_choices)

        
            all_choices = re.findall(r'[A-E]', response)
            if all_choices:
                unique_choices = sorted(list(set(all_choices)))
                return ''.join(unique_choices)
            
            
            return choices[random.randint(0,4)]

        extracted_preds = [extract_choice(p) for p in predictions]

        # Subtask statistics
        if data_samples is not None:
            subtask_stats = defaultdict(lambda: {"correct":0, "total":0})
            for p, g, s in zip(extracted_preds, ground_truths, data_samples):
                subtask = s.get('benchmark', 'unknown')
                if p == g:
                    subtask_stats[subtask]["correct"] += 1
                subtask_stats[subtask]["total"] += 1
            subtask_acc = {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k,v in subtask_stats.items()}
        else:
            subtask_acc = {}

        correct = sum(1 for p, g in zip(extracted_preds, ground_truths) if set(p) == set(g))
        is_correct_list = [set(p) == set(g) for p, g in zip(extracted_preds, ground_truths)]
        total = len(ground_truths)
        accuracy = correct / total if total else 0.0

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_rate": 1.0 - accuracy,
            "is_correct_list": is_correct_list,
            "subtask_accuracy": subtask_acc,
        }

        return result