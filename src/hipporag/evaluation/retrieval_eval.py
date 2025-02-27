from typing import List, Tuple, Dict, Any, Optional
import numpy as np


from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig




logger = get_logger(__name__)



class RetrievalRecall(BaseMetric):
    
    metric_name: str = "retrieval_recall"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Recall@k for each example and pools results for all queries.

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
            retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.
            k_list (List[int]): List of k values to calculate Recall@k for.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A pooled dictionary with the averaged Recall@k across all examples.
                - A list of dictionaries with Recall@k for each example.
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}
  
            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                # Calculate intersection with gold documents
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0
            
            # Append example results
            example_eval_results.append(example_eval_result)
            
            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results