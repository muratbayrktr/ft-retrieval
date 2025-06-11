import pyterrier as pt
from typing import Dict
import pandas as pd
from logger import get_logger

logger = get_logger("evaluator")

class Evaluator:
    def __init__(self, k: int = 10):
        """
        Initialize the evaluator with specified k value.
        
        Args:
            k (int): Number of top results to consider for evaluation
        """
        self.k = k
        
    def evaluate(self, results: pd.DataFrame, qrels: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate retrieval results using NDCG and MAP.
        
        Args:
            results: Retrieval results DataFrame
            qrels: Relevance judgments DataFrame
            
        Returns:
            Dictionary containing NDCG and MAP scores
        """
        logger.info("Starting evaluation...")
        
        # Ensure qid columns are strings in both DataFrames
        results['qid'] = results['qid'].astype(str)
        qrels['qid'] = qrels['qid'].astype(str)
        
        # Create evaluation measures
        eval_metrics = [
            pt.measures.NDCG @ self.k,
            pt.measures.MAP @ self.k,
            pt.measures.P @ self.k,
            pt.measures.R @ self.k,
        ]
        
        # Use the direct evaluation approach
        scores = pt.Utils.evaluate(results, qrels, metrics=eval_metrics)
        
        # Extract scores
        ndcg_key = f"nDCG@{self.k}"
        map_key = f"AP@{self.k}"
        p_key = f"P@{self.k}"
        r_key = f"R@{self.k}"
        
        ndcg_score = scores.get(ndcg_key, 0.0)
        map_score = scores.get(map_key, 0.0)
        p_score = scores.get(p_key, 0.0)
        r_score = scores.get(r_key, 0.0)
        
        logger.info(f"""Evaluation completed.
            NDCG@{self.k}: {ndcg_score:.4f},
            MAP@{self.k}: {map_score:.4f},
            P@{self.k}: {p_score:.4f},
            R@{self.k}: {r_score:.4f}
""")
        
        return {
            'ndcg': ndcg_score,
            'map': map_score,
            'p': p_score,
            'r': r_score
        } 