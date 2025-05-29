import pyterrier as pt
from typing import Optional
import pandas as pd
from logger import get_logger

logger = get_logger("retriever")

class Retriever:
    def __init__(self, index_ref):
        """
        Initialize the retriever with an index reference.
        
        Args:
            index_ref: PyTerrier index reference
        """
        self.bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True)
        self.results = None
        
    def retrieve(self, queries: pd.DataFrame) -> pd.DataFrame:
        """
        Perform retrieval using BM25.
        
        Args:
            queries: Query DataFrame
            
        Returns:
            pd.DataFrame: Retrieval results
        """
        logger.info(f"Starting retrieval for {len(queries)} queries...")
        self.results = self.bm25.transform(queries)
        logger.info(f"Retrieval completed. Retrieved {len(self.results)} results.")
        return self.results
        
    def get_results(self):
        """Return the retrieval results."""
        if self.results is None:
            raise ValueError("No retrieval performed yet. Call retrieve first.")
        return self.results 