import pyterrier as pt
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Callable, Optional, Any
from collections import defaultdict, Counter
from logger import get_logger

logger = get_logger("custom_retriever")

class CustomRetriever:
    """
    A flexible custom retriever that provides custom scoring algorithms.
    Uses PyTerrier's BM25 as a base and applies custom modifications.
    """
    
    def __init__(self, index_ref, verbose: bool = True):
        """
        Initialize the custom retriever.
        
        Args:
            index_ref: PyTerrier index reference
            verbose (bool): Whether to show progress information
        """
        self.index_ref = index_ref
        self.verbose = verbose
        
        # Base retriever for getting candidates - use old API to avoid query parsing issues
        try:
            self.base_retriever = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=False)
        except:
            # Fallback to newer API
            self.base_retriever = pt.terrier.Retriever(index_ref, wmodel="BM25")
        
        # Get index statistics
        self.index_stats = self._get_index_stats()
        logger.info(f"Loaded index with {self.index_stats['num_docs']} documents")
        
        # Available scoring algorithms
        self.scoring_algorithms = {
            'tf_idf': self._score_tf_idf_modifier,
            'bm25': self._score_bm25_modifier,
            'bm25_plus': self._score_bm25_plus_modifier,
            'cosine': self._score_cosine_modifier,
            'language_model': self._score_language_model_modifier,
            'financial_boost': self._score_financial_boost,  # Domain-specific algorithm
            'reverse_test': self._score_reverse_test,  # Test algorithm that reverses ranking
            'random_test': self._score_random_test,   # Test algorithm with random scoring
            'custom': None  # Will be set by user
        }
        
        # Current scoring algorithm
        self.current_algorithm = 'bm25'
        
        # BM25 parameters (can be tuned)
        self.k1 = 1.2
        self.b = 0.75
        
        # Language model smoothing parameter
        self.mu = 2000
        
        # Custom scoring function
        self.custom_scorer: Optional[Callable] = None
        
    def _get_index_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the index."""
        try:
            index = self.index_ref
            stats = {
                'num_docs': index.getCollectionStatistics().getNumberOfDocuments(),
                'num_terms': index.getCollectionStatistics().getNumberOfUniqueTerms(),
                'num_tokens': index.getCollectionStatistics().getNumberOfTokens(),
                'avg_doc_length': index.getCollectionStatistics().getAverageDocumentLength()
            }
            return stats
        except Exception as e:
            logger.warning(f"Could not get index statistics: {e}")
            return {'num_docs': 100000, 'num_terms': 50000, 'num_tokens': 5000000, 'avg_doc_length': 50}
    
    def set_scoring_algorithm(self, algorithm: str, **params):
        """
        Set the scoring algorithm to use.
        
        Args:
            algorithm (str): Name of the algorithm ('tf_idf', 'bm25', 'bm25_plus', 'cosine', 'language_model', 'custom')
            **params: Algorithm-specific parameters
        """
        if algorithm not in self.scoring_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.scoring_algorithms.keys())}")
        
        self.current_algorithm = algorithm
        
        # Set algorithm-specific parameters
        if algorithm == 'bm25' or algorithm == 'bm25_plus':
            self.k1 = params.get('k1', 1.2)
            self.b = params.get('b', 0.75)
            logger.info(f"Set {algorithm} parameters: k1={self.k1}, b={self.b}")
        elif algorithm == 'language_model':
            self.mu = params.get('mu', 2000)
            logger.info(f"Set language model parameter: mu={self.mu}")
    
    def set_custom_scorer(self, scorer_func: Callable):
        """
        Set a custom scoring function.
        
        Args:
            scorer_func: Function with signature (row, query_terms) -> score
                where row contains 'qid', 'docno', 'rank', 'score'
        """
        self.custom_scorer = scorer_func
        self.current_algorithm = 'custom'
        logger.info("Set custom scoring function")
    
    def _score_tf_idf_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """TF-IDF style modifier with significant changes."""
        original_score = row['score']
        
        # More aggressive query length modifications
        query_length = len(query_terms)
        if query_length > 8:
            boost = 1.5  # 50% boost for very long queries
        elif query_length > 5:
            boost = 1.3  # 30% boost for long queries
        elif query_length < 3:
            boost = 0.6  # 40% penalty for very short queries
        else:
            boost = 1.0
        
        # Additional boost for document position (simulate TF-IDF preferences)
        rank_penalty = 1.0 - (row.get('rank', 0) * 0.001)  # Slight penalty for lower ranks
        
        return original_score * boost * rank_penalty
    
    def _score_bm25_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Enhanced BM25 with significant query-dependent modifications."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        boost = 1.0
        
        # Strong boost for question-type queries
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(word in query_text for word in question_words):
            boost *= 1.4  # 40% boost for questions
        
        # Boost for specific query patterns
        if 'identify' in query_text:
            boost *= 1.2  # 20% boost for identification queries
        
        # Penalty for very common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        common_count = sum(1 for word in common_words if word in query_text)
        if common_count > 3:
            boost *= 0.8  # Penalty for queries with many common words
        
        return original_score * boost
    
    def _score_bm25_plus_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """BM25+ with rank-dependent scoring."""
        original_score = row['score']
        
        # Strong rank-dependent boost (BM25+ philosophy)
        rank = row.get('rank', 0)
        if rank < 10:
            rank_boost = 2.0  # Double score for top 10
        elif rank < 50:
            rank_boost = 1.5  # 50% boost for top 50
        elif rank < 100:
            rank_boost = 1.2  # 20% boost for top 100
        else:
            rank_boost = 0.9  # Slight penalty for lower ranks
        
        # Additional constant boost (BM25+ characteristic)
        constant_boost = 0.5
        
        return (original_score * rank_boost) + constant_boost
    
    def _score_cosine_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Cosine-inspired scoring with document length normalization."""
        original_score = row['score']
        docno = row['docno']
        
        # Aggressive length-based modifications
        try:
            # Extract year and document number for length estimation
            if 'FT' in docno:
                # Extract document number (last part after -)
                parts = docno.split('-')
                if len(parts) > 1:
                    doc_num = int(parts[-1])
                    # Assume higher numbers = longer documents
                    if doc_num > 5000:
                        length_factor = 0.7  # Strong penalty for very long docs
                    elif doc_num > 2000:
                        length_factor = 0.85  # Moderate penalty
                    elif doc_num < 500:
                        length_factor = 1.3  # Boost short documents
                    else:
                        length_factor = 1.0
                else:
                    length_factor = 1.0
            else:
                length_factor = 1.0
        except:
            length_factor = 1.0
        
        # Query length interaction
        query_length = len(query_terms)
        if query_length > 6:
            query_factor = 1.2  # Boost for longer queries (more specific)
        else:
            query_factor = 1.0
        
        return original_score * length_factor * query_factor
    
    def _score_language_model_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Language model with strong domain-specific boosting."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        
        # Strong domain-specific boosting
        financial_terms = {'financial', 'economic', 'market', 'business', 'trade', 'bank', 'money', 'investment'}
        tech_terms = {'technology', 'computer', 'internet', 'digital', 'software', 'hardware'}
        medical_terms = {'health', 'medical', 'disease', 'treatment', 'drug', 'medicine', 'hospital'}
        
        boost = 1.0
        
        # Financial domain (strongest boost since we have FT collection)
        financial_matches = sum(1 for term in financial_terms if term in query_text)
        if financial_matches > 0:
            boost *= (1.0 + 0.3 * financial_matches)  # 30% boost per financial term
        
        # Tech domain
        tech_matches = sum(1 for term in tech_terms if term in query_text)
        if tech_matches > 0:
            boost *= (1.0 + 0.2 * tech_matches)  # 20% boost per tech term
        
        # Medical domain
        medical_matches = sum(1 for term in medical_terms if term in query_text)
        if medical_matches > 0:
            boost *= (1.0 + 0.25 * medical_matches)  # 25% boost per medical term
        
        # Document type boost (FT documents get extra boost for financial queries)
        docno = row['docno']
        if 'FT' in docno and financial_matches > 0:
            boost *= 1.4  # Extra 40% boost for FT docs with financial queries
        
        # Query specificity boost
        if len(query_terms) > 7:
            boost *= 1.2  # Boost very specific queries
        
        return original_score * boost
    
    def _score_reverse_test(self, row: pd.Series, query_terms: List[str]) -> float:
        """Test algorithm that reverses the original ranking order."""
        original_score = row['score']
        rank = row.get('rank', 0)
        
        # Get max rank to properly reverse
        max_rank = 1000  # Assume max 1000 results
        
        # Completely reverse the ranking: lowest ranked docs get highest scores
        reversed_score = max_rank - rank
        
        return float(reversed_score)
    
    def _score_random_test(self, row: pd.Series, query_terms: List[str]) -> float:
        """Test algorithm with random scoring to verify system is working."""
        import random
        
        # Use document ID as seed for reproducible "randomness"
        docno = row['docno']
        seed = hash(docno) % 1000000
        random.seed(seed)
        
        # Return random score between 0 and 100
        return random.uniform(0, 100)
    
    def _score_financial_boost(self, row: pd.Series, query_terms: List[str]) -> float:
        """Financial domain boost algorithm that should improve results."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        docno = row['docno']
        
        # Strong boost for financial terms in query
        financial_terms = {'financial', 'economic', 'market', 'business', 'trade', 'bank', 'money', 'investment', 'company', 'corporate'}
        financial_matches = sum(1 for term in financial_terms if term in query_text)
        
        if financial_matches > 0:
            # Very strong boost for FT documents with financial queries
            if 'FT' in docno:
                boost = 2.0 + (financial_matches * 0.5)  # 2x base + 50% per financial term
            else:
                boost = 1.5 + (financial_matches * 0.3)  # 1.5x base + 30% per term
        else:
            # Boost question-type queries
            if any(word in query_text for word in ['what', 'how', 'identify', 'find']):
                boost = 1.2
            else:
                boost = 1.0
        
        return original_score * boost
    
    def retrieve(self, queries: pd.DataFrame, num_results: int = 1000) -> pd.DataFrame:
        """
        Perform retrieval using the current scoring algorithm.
        
        Args:
            queries: DataFrame with columns 'qid' and 'query'
            num_results (int): Number of results to return per query
            
        Returns:
            DataFrame with columns 'qid', 'docno', 'rank', 'score'
        """
        logger.info(f"Starting custom retrieval for {len(queries)} queries using {self.current_algorithm} algorithm")
        
        # First, get base results from BM25
        try:
            base_results = self.base_retriever.transform(queries)
        except Exception as e:
            logger.error(f"Error in base retrieval: {e}")
            return pd.DataFrame(columns=['qid', 'docno', 'rank', 'score'])
        
        if len(base_results) == 0:
            logger.warning("No base results from BM25")
            return pd.DataFrame(columns=['qid', 'docno', 'rank', 'score'])
        
        # Apply custom scoring
        enhanced_results = []
        
        for _, query_row in queries.iterrows():
            qid = str(query_row['qid'])
            query_text = str(query_row['query'])
            query_terms = query_text.lower().split()
            
            # Get results for this query
            query_results = base_results[base_results['qid'] == qid].copy()
            
            if len(query_results) == 0:
                continue
            
            # Apply custom scoring
            for idx, result_row in query_results.iterrows():
                # Get new score
                if self.current_algorithm == 'custom' and self.custom_scorer:
                    new_score = self.custom_scorer(result_row, query_terms)
                else:
                    scorer_func = self.scoring_algorithms[self.current_algorithm]
                    new_score = scorer_func(result_row, query_terms)
                
                enhanced_results.append({
                    'qid': result_row['qid'],
                    'docno': result_row['docno'],
                    'score': new_score
                })
        
        # Convert to DataFrame and re-rank
        if enhanced_results:
            results_df = pd.DataFrame(enhanced_results)
            
            # Re-rank within each query
            final_results = []
            for qid in results_df['qid'].unique():
                query_results = results_df[results_df['qid'] == qid].copy()
                query_results = query_results.sort_values('score', ascending=False)
                query_results['rank'] = range(len(query_results))
                
                # Limit to requested number of results
                query_results = query_results.head(num_results)
                final_results.append(query_results)
            
            final_df = pd.concat(final_results, ignore_index=True)
            logger.info(f"Custom retrieval completed. Retrieved {len(final_df)} results.")
            return final_df
        else:
            return pd.DataFrame(columns=['qid', 'docno', 'rank', 'score'])
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the current algorithm and parameters."""
        info = {
            'algorithm': self.current_algorithm,
            'available_algorithms': list(self.scoring_algorithms.keys())
        }
        
        if self.current_algorithm in ['bm25', 'bm25_plus']:
            info.update({'k1': self.k1, 'b': self.b})
        elif self.current_algorithm == 'language_model':
            info.update({'mu': self.mu})
        elif self.current_algorithm == 'custom':
            info.update({'custom_scorer': self.custom_scorer is not None})
        
        return info 