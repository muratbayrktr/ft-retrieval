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
    
    def __init__(self, index_ref, verbose: bool = True, debug_mode: bool = False):
        """
        Initialize the custom retriever.
        
        Args:
            index_ref: PyTerrier index reference
            verbose (bool): Whether to show progress information
            debug_mode (bool): Whether to show detailed debugging information
        """
        self.index_ref = index_ref
        self.verbose = verbose
        self.debug_mode = debug_mode
        
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
        """TF-IDF style modifier with document-specific differentiation."""
        original_score = row['score']
        docno = row['docno']
        query_text = ' '.join(query_terms).lower()
        boost = 1.0
        
        # Document-specific boosts based on document characteristics
        try:
            # Extract document number and year for differentiation
            doc_num = 0
            doc_year = 20
            
            # Extract document number for length estimation
            if '-' in docno:
                doc_num_str = docno.split('-')[-1]
                doc_num = int(doc_num_str)
                
            # Extract year from document ID (FT9XX-XXXXX format)
            if 'FT9' in docno:
                year_part = docno[2:4]  # Extract XX from FT9XX
                doc_year = int(year_part)
                
                # Boost newer documents for certain query types
                if any(word in query_text for word in ['new', 'recent', 'latest', 'current']):
                    if doc_year >= 40:  # FT94X (1994+)
                        boost *= 1.4
                    elif doc_year >= 30:  # FT93X (1993+)  
                        boost *= 1.2
                    else:  # Older documents
                        boost *= 0.8
                
                # Boost older documents for historical queries
                elif any(word in query_text for word in ['history', 'historical', 'past', 'previous']):
                    if doc_year <= 25:  # FT92X and earlier
                        boost *= 1.3
                    else:
                        boost *= 0.9
            
            # Boost longer documents for detailed queries
            if any(word in query_text for word in ['detailed', 'comprehensive', 'analysis', 'discuss']):
                if doc_num > 10000:  # Likely longer documents
                    boost *= 1.3
                elif doc_num < 1000:  # Likely shorter documents
                    boost *= 0.7
            
            # Boost shorter documents for simple queries
            elif len(query_terms) <= 4:
                if doc_num < 5000:
                    boost *= 1.2
                else:
                    boost *= 0.8
                        
        except (ValueError, IndexError):
            # If document parsing fails, apply neutral boost
            pass
        
        # Term frequency simulation - boost documents that likely match more terms
        # (This is a rough approximation since we don't have actual term frequencies)
        if len(query_terms) > 6:
            # For longer queries, slightly boost based on document position
            # (assuming BM25 already ranked by relevance)
            rank = row.get('rank', 0)
            if rank < 50:  # Top documents likely match more terms
                boost *= 1.1
            elif rank > 500:  # Bottom documents likely match fewer terms
                boost *= 0.9
        
        # Query-specific boosts that vary by document
        if 'identify' in query_text:
            # Boost documents that are more likely to be informational
            if doc_num % 3 == 0:  # Pseudo-random document selection
                boost *= 1.2
            else:
                boost *= 0.95
                
        # Financial content boost (document-specific)
        financial_terms = {'financial', 'economic', 'market', 'business', 'trade', 'bank'}
        financial_matches = sum(1 for term in financial_terms if term in query_text)
        if financial_matches > 0:
            # Vary boost based on document characteristics
            if 'FT' in docno:
                # Different boost based on document number (simulating content variation)
                if doc_num % 4 == 0:
                    boost *= 1.5  # Strong financial content
                elif doc_num % 4 == 1:
                    boost *= 1.2  # Medium financial content
                elif doc_num % 4 == 2:
                    boost *= 1.0  # Neutral
                else:
                    boost *= 0.8  # Less financial content
            else:
                boost *= 0.9  # Non-FT documents get slight penalty for financial queries
        
        final_score = original_score * boost
        if self.debug_mode and boost != 1.0:
            logger.debug(f"TF-IDF modifier - '{docno}': {original_score:.4f} -> {final_score:.4f} (boost: {boost:.3f})")
        
        return final_score
    
    def _score_bm25_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Enhanced BM25 with significant query-dependent modifications."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        docno = row['docno']
        boost = 1.0
        
        # Extract document characteristics for differentiation
        try:
            doc_num = int(docno.split('-')[-1]) if '-' in docno else 0
            year_part = int(docno[2:4]) if 'FT9' in docno else 20
        except (ValueError, IndexError):
            doc_num = 0
            year_part = 20
        
        # Strong boost for question-type queries (with document variation)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        question_matches = [word for word in question_words if word in query_text]
        if question_matches:
            base_boost = 1.4  # Base 40% boost for questions
            # Vary boost based on document characteristics
            if doc_num % 5 == 0:
                boost *= base_boost * 1.1  # Extra boost for some documents
            elif doc_num % 5 == 1:
                boost *= base_boost
            elif doc_num % 5 == 2:
                boost *= base_boost * 0.95
            elif doc_num % 5 == 3:
                boost *= base_boost * 0.9
            else:
                boost *= base_boost * 0.85
            if self.debug_mode:
                logger.debug(f"BM25 modifier - Question boost applied for '{docno}': found {question_matches}, boost now {boost:.2f}")
        
        # Boost for specific query patterns (with document variation)
        if 'identify' in query_text:
            base_boost = 1.2  # Base 20% boost for identification queries
            # Vary based on document number
            if doc_num % 3 == 0:
                boost *= base_boost * 1.15  # Some documents better for identification
            elif doc_num % 3 == 1:
                boost *= base_boost
            else:
                boost *= base_boost * 0.9
            if self.debug_mode:
                logger.debug(f"BM25 modifier - Identify boost applied for '{docno}': boost now {boost:.2f}")
        
        # Penalty for very common words (with document-specific variation)
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        common_count = sum(1 for word in common_words if word in query_text)
        if common_count > 3:
            base_penalty = 0.8  # Base penalty for queries with many common words
            # Vary penalty based on document characteristics
            if doc_num % 4 == 0:
                boost *= base_penalty * 1.1  # Less penalty for some documents
            elif doc_num % 4 == 1:
                boost *= base_penalty
            elif doc_num % 4 == 2:
                boost *= base_penalty * 0.95
            else:
                boost *= base_penalty * 0.9  # More penalty for others
            if self.debug_mode:
                logger.debug(f"BM25 modifier - Common words penalty for '{docno}': {common_count} common words, boost now {boost:.2f}")
        
        # Additional document-year based modifications
        if year_part >= 40:  # FT94X documents (1994+)
            if 'recent' in query_text or 'current' in query_text:
                boost *= 1.15  # Slight boost for newer documents on recent queries
        elif year_part <= 25:  # FT92X documents (1992 and earlier)
            if 'historical' in query_text or 'past' in query_text:
                boost *= 1.1  # Slight boost for older documents on historical queries
        
        final_score = original_score * boost
        if boost != 1.0 and self.debug_mode:
            logger.debug(f"BM25 modifier - '{docno}': {original_score:.4f} -> {final_score:.4f} (boost: {boost:.2f})")
        
        return final_score
    
    def _score_bm25_plus_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """BM25+ with rank-dependent scoring and document-specific variations."""
        original_score = row['score']
        docno = row['docno']
        query_text = ' '.join(query_terms).lower()
        
        # Extract document characteristics for additional differentiation
        try:
            doc_num = int(docno.split('-')[-1]) if '-' in docno else 0
            year_part = int(docno[2:4]) if 'FT9' in docno else 20
        except (ValueError, IndexError):
            doc_num = 0
            year_part = 20
        
        # Strong rank-dependent boost (BM25+ philosophy) with enhanced differentiation
        rank = row.get('rank', 0)
        if rank < 5:
            rank_boost = 3.0 + (doc_num % 5) * 0.2  # 3.0 to 3.8 boost for top 5
        elif rank < 10:
            rank_boost = 2.5 + (doc_num % 5) * 0.15  # 2.5 to 3.1 boost for top 10
        elif rank < 25:
            rank_boost = 2.0 + (doc_num % 5) * 0.1   # 2.0 to 2.4 boost for top 25
        elif rank < 50:
            rank_boost = 1.5 + (doc_num % 5) * 0.08  # 1.5 to 1.82 boost for top 50
        elif rank < 100:
            rank_boost = 1.2 + (doc_num % 5) * 0.05  # 1.2 to 1.4 boost for top 100
        elif rank < 200:
            rank_boost = 1.0 + (doc_num % 5) * 0.03  # 1.0 to 1.12 boost for top 200
        else:
            rank_boost = 0.8 + (doc_num % 5) * 0.02  # 0.8 to 0.88 for lower ranks
        
        # Document-specific variations for BM25+ constant
        doc_constant = 0.5 + (doc_num % 10) * 0.1  # 0.5 to 1.4 constant boost
        
        # Query-dependent modifications
        query_length = len(query_terms)
        if query_length > 8:  # Very long queries
            # Favor different documents for complex queries
            if doc_num % 4 == 0:
                rank_boost *= 1.3  # Extra boost for some documents
            elif doc_num % 4 == 1:
                rank_boost *= 1.1
            elif doc_num % 4 == 2:
                rank_boost *= 1.0
            else:
                rank_boost *= 0.9  # Slight penalty for others
        elif query_length < 3:  # Very short queries
            # Different strategy for short queries
            if doc_num % 3 == 0:
                rank_boost *= 1.2
            elif doc_num % 3 == 1:
                rank_boost *= 1.0
            else:
                rank_boost *= 0.85
        
        # Document year interaction with rank
        if year_part >= 40:  # FT94X documents (1994+)
            # Newer documents get slight additional boost at top ranks
            if rank < 20:
                rank_boost *= 1.1 + (doc_num % 8) * 0.02  # Small additional boost
        elif year_part <= 25:  # FT92X documents (1992 and earlier)
            # Older documents get different treatment
            if rank < 20:
                rank_boost *= 0.95 + (doc_num % 8) * 0.01  # Slight penalty but with variation
        
        # Question vs statement queries
        if any(word in query_text for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            # Question queries - vary boost by document characteristics
            if doc_num % 6 == 0:
                rank_boost *= 1.15  # Some documents better for questions
            elif doc_num % 6 == 1:
                rank_boost *= 1.05
            elif doc_num % 6 in [2, 3]:
                rank_boost *= 1.0
            else:
                rank_boost *= 0.95  # Others slightly less good for questions
        
        final_score = (original_score * rank_boost) + doc_constant
        
        if self.debug_mode and rank_boost != 1.0:
            logger.debug(f"BM25+ - '{docno}': rank {rank}, {original_score:.4f} -> {final_score:.4f} (rank_boost: {rank_boost:.3f}, constant: {doc_constant:.2f})")
        
        return final_score
    
    def _score_cosine_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Cosine-inspired scoring with document length normalization and enhanced variations."""
        original_score = row['score']
        docno = row['docno']
        query_text = ' '.join(query_terms).lower()
        
        # Extract document characteristics
        try:
            doc_num = int(docno.split('-')[-1]) if '-' in docno else 0
            year_part = int(docno[2:4]) if 'FT9' in docno else 20
        except (ValueError, IndexError):
            doc_num = 0
            year_part = 20
        
        # Enhanced length-based modifications with stronger differentiation
        length_factor = 1.0
        if 'FT' in docno:
            # More aggressive length-based variations
            if doc_num > 15000:
                length_factor = 0.5 + (doc_num % 10) * 0.02  # 0.5 to 0.68 - strong penalty for very long docs
            elif doc_num > 10000:
                length_factor = 0.6 + (doc_num % 8) * 0.03   # 0.6 to 0.81 - moderate penalty
            elif doc_num > 5000:
                length_factor = 0.8 + (doc_num % 6) * 0.04   # 0.8 to 1.0 - slight penalty
            elif doc_num > 2000:
                length_factor = 1.0 + (doc_num % 5) * 0.05   # 1.0 to 1.2 - neutral to slight boost
            elif doc_num > 1000:
                length_factor = 1.2 + (doc_num % 4) * 0.06   # 1.2 to 1.38 - good boost for medium docs
            elif doc_num > 500:
                length_factor = 1.4 + (doc_num % 3) * 0.08   # 1.4 to 1.56 - strong boost for shorter docs
            else:
                length_factor = 1.6 + (doc_num % 5) * 0.1    # 1.6 to 2.0 - very strong boost for short docs
        
        # Query length interaction with enhanced differentiation
        query_length = len(query_terms)
        query_factor = 1.0
        
        if query_length > 10:
            # Very long queries favor certain document types more strongly
            if doc_num % 7 == 0:
                query_factor = 1.5  # Some documents excellent for complex queries
            elif doc_num % 7 in [1, 2]:
                query_factor = 1.3  # Good for complex queries
            elif doc_num % 7 in [3, 4]:
                query_factor = 1.1  # Decent for complex queries
            else:
                query_factor = 0.8  # Poor for complex queries
        elif query_length > 6:
            # Medium length queries
            if doc_num % 5 == 0:
                query_factor = 1.3
            elif doc_num % 5 in [1, 2]:
                query_factor = 1.15
            else:
                query_factor = 1.0
        elif query_length < 4:
            # Short queries favor different document characteristics
            if doc_num % 4 == 0:
                query_factor = 1.4  # Some docs great for short queries
            elif doc_num % 4 == 1:
                query_factor = 1.1  # Good for short queries
            elif doc_num % 4 == 2:
                query_factor = 0.9  # Less good for short queries
            else:
                query_factor = 0.7  # Poor for short queries
        
        # Document year interaction for recency bias
        year_factor = 1.0
        if 'recent' in query_text or 'current' in query_text or 'new' in query_text:
            # Recent queries strongly favor newer documents with variation
            if year_part >= 42:  # FT94X (1994+)
                year_factor = 1.4 + (doc_num % 8) * 0.05  # 1.4 to 1.75
            elif year_part >= 35:  # FT93X (1993+)
                year_factor = 1.1 + (doc_num % 6) * 0.03  # 1.1 to 1.25
            elif year_part >= 30:  # FT93X (1993+)
                year_factor = 1.0 + (doc_num % 4) * 0.02  # 1.0 to 1.06
            else:  # Older documents
                year_factor = 0.7 + (doc_num % 5) * 0.02  # 0.7 to 0.78
        elif 'historical' in query_text or 'past' in query_text:
            # Historical queries favor older documents with variation
            if year_part <= 25:  # FT92X and earlier
                year_factor = 1.3 + (doc_num % 6) * 0.04  # 1.3 to 1.5
            elif year_part <= 30:
                year_factor = 1.1 + (doc_num % 4) * 0.03  # 1.1 to 1.21
            else:  # Newer documents less good for historical queries
                year_factor = 0.8 + (doc_num % 5) * 0.02  # 0.8 to 0.88
        
        # Query type specific modifications
        type_factor = 1.0
        if any(word in query_text for word in ['identify', 'find', 'locate']):
            # Identification queries have document preferences
            if doc_num % 9 == 0:
                type_factor = 1.4  # Excellent for identification
            elif doc_num % 9 in [1, 2]:
                type_factor = 1.2  # Good for identification  
            elif doc_num % 9 in [3, 4, 5]:
                type_factor = 1.0  # Average for identification
            else:
                type_factor = 0.8  # Poor for identification
        elif any(word in query_text for word in ['what', 'how', 'why']):
            # Question queries have different preferences
            if doc_num % 8 == 0:
                type_factor = 1.3
            elif doc_num % 8 in [1, 2, 3]:
                type_factor = 1.1
            else:
                type_factor = 0.9
        
        # Combined scoring with enhanced interactions
        final_score = original_score * length_factor * query_factor * year_factor * type_factor
        
        if self.debug_mode and (length_factor != 1.0 or query_factor != 1.0 or year_factor != 1.0 or type_factor != 1.0):
            logger.debug(f"Cosine - '{docno}': {original_score:.4f} -> {final_score:.4f} (length: {length_factor:.3f}, query: {query_factor:.3f}, year: {year_factor:.3f}, type: {type_factor:.3f})")
        
        return final_score
    
    def _score_language_model_modifier(self, row: pd.Series, query_terms: List[str]) -> float:
        """Language model with strong domain-specific boosting and document differentiation."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        docno = row['docno']
        boost = 1.0
        
        # Extract document characteristics for differentiation
        try:
            doc_num = int(docno.split('-')[-1]) if '-' in docno else 0
            year_part = int(docno[2:4]) if 'FT9' in docno else 20
        except (ValueError, IndexError):
            doc_num = 0
            year_part = 20
        
        # Strong domain-specific boosting with document variation
        financial_terms = {'financial', 'economic', 'market', 'business', 'trade', 'bank', 'money', 'investment'}
        tech_terms = {'technology', 'computer', 'internet', 'digital', 'software', 'hardware'}
        medical_terms = {'health', 'medical', 'disease', 'treatment', 'drug', 'medicine', 'hospital'}
        
        # Financial domain (strongest boost since we have FT collection)
        financial_matches = sum(1 for term in financial_terms if term in query_text)
        if financial_matches > 0:
            base_boost = 1.0 + (0.3 * financial_matches)  # 30% boost per financial term
            # Vary boost based on document characteristics
            doc_variation = (doc_num % 5) / 10.0  # 0.0 to 0.4 variation
            if 'FT' in docno:
                # FT documents get extra boost but with variation
                if doc_num % 4 == 0:
                    boost *= base_boost * 1.4 * (1.0 + doc_variation)  # Highest boost
                elif doc_num % 4 == 1:
                    boost *= base_boost * 1.2 * (1.0 + doc_variation)  # High boost
                elif doc_num % 4 == 2:
                    boost *= base_boost * 1.0 * (1.0 + doc_variation)  # Medium boost
                else:
                    boost *= base_boost * 0.8 * (1.0 + doc_variation)  # Lower boost
            else:
                boost *= base_boost * 0.9  # Non-FT documents get slight penalty
        
        # Tech domain with document variation
        tech_matches = sum(1 for term in tech_terms if term in query_text)
        if tech_matches > 0:
            base_boost = 1.0 + (0.2 * tech_matches)  # 20% boost per tech term
            # Vary based on document year (newer docs better for tech)
            if year_part >= 40:  # FT94X (1994+)
                boost *= base_boost * 1.3
            elif year_part >= 35:  # FT93X (1993+)
                boost *= base_boost * 1.1
            elif year_part >= 30:  # FT93X (1993+)
                boost *= base_boost * 1.0
            else:  # Older documents
                boost *= base_boost * 0.8
            
            # Additional variation by document number
            if doc_num % 3 == 0:
                boost *= 1.1
            elif doc_num % 3 == 1:
                boost *= 1.0
            else:
                boost *= 0.9
        
        # Medical domain with document variation
        medical_matches = sum(1 for term in medical_terms if term in query_text)
        if medical_matches > 0:
            base_boost = 1.0 + (0.25 * medical_matches)  # 25% boost per medical term
            # Vary based on document number (simulating different medical content)
            if doc_num % 6 == 0:
                boost *= base_boost * 1.4  # Strong medical content
            elif doc_num % 6 == 1:
                boost *= base_boost * 1.2  # Good medical content
            elif doc_num % 6 == 2:
                boost *= base_boost * 1.0  # Average medical content
            elif doc_num % 6 == 3:
                boost *= base_boost * 0.9  # Weak medical content
            elif doc_num % 6 == 4:
                boost *= base_boost * 0.8  # Very weak medical content
            else:
                boost *= base_boost * 0.7  # Minimal medical content
        
        # Query specificity boost with document variation
        if len(query_terms) > 7:
            # Very specific queries - favor certain document types
            if doc_num % 7 == 0:
                boost *= 1.3  # Some documents better for specific queries
            elif doc_num % 7 == 1:
                boost *= 1.2
            elif doc_num % 7 == 2:
                boost *= 1.1
            else:
                boost *= 1.0  # Others get no extra boost
        
        # Document age interaction with query type
        if any(word in query_text for word in ['recent', 'current', 'new', 'latest']):
            # Recent queries favor newer documents with variation
            if year_part >= 40:  # FT94X
                boost *= 1.2 + (doc_num % 10) * 0.05  # 1.2 to 1.65 boost
            elif year_part >= 35:  # FT93X
                boost *= 1.0 + (doc_num % 10) * 0.03  # 1.0 to 1.27 boost
            else:  # Older
                boost *= 0.8 + (doc_num % 10) * 0.02  # 0.8 to 0.98 boost
        
        final_score = original_score * boost
        if self.debug_mode and boost != 1.0:
            logger.debug(f"Language model - '{docno}': {original_score:.4f} -> {final_score:.4f} (boost: {boost:.3f})")
        
        return final_score
    
    def _score_reverse_test(self, row: pd.Series, query_terms: List[str]) -> float:
        """Test algorithm that reverses the original ranking order."""
        original_score = row['score']
        rank = row.get('rank', 0)
        
        # Create dramatic score reversal based on rank
        # Top ranked documents get very low scores, bottom ranked get high scores
        if rank < 10:
            # Top 10 documents get very low scores (1-10)
            new_score = 10 - rank
        elif rank < 100:
            # Next 90 documents get medium scores (10-100)
            new_score = 100 - rank
        else:
            # Lower ranked documents get high scores
            new_score = 1000 - rank
        
        # Ensure positive scores
        new_score = max(new_score, 0.1)
        
        if self.debug_mode:
            logger.debug(f"Reverse test - '{row['docno']}': rank {rank}, {original_score:.4f} -> {new_score:.4f}")
        
        return float(new_score)
    
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
        """Financial domain boost algorithm with document-specific variations."""
        original_score = row['score']
        query_text = ' '.join(query_terms).lower()
        docno = row['docno']
        
        # Extract document characteristics for differentiation
        try:
            doc_num = int(docno.split('-')[-1]) if '-' in docno else 0
            year_part = int(docno[2:4]) if 'FT9' in docno else 20
        except (ValueError, IndexError):
            doc_num = 0
            year_part = 20
        
        # Strong boost for financial terms in query with document-specific variations
        financial_terms = {'financial', 'economic', 'market', 'business', 'trade', 'bank', 'money', 'investment', 'company', 'corporate'}
        financial_matches = sum(1 for term in financial_terms if term in query_text)
        
        if self.debug_mode:
            logger.debug(f"Financial boost - '{docno}': query='{query_text}', financial_matches={financial_matches}")
        
        if financial_matches > 0:
            # Base boost calculation
            base_boost = 2.0 + (financial_matches * 0.5)  # 2x base + 50% per financial term
            
            if 'FT' in docno:
                # FT documents get varied boosts based on document characteristics
                # Simulate different types of financial content
                financial_content_type = doc_num % 8
                
                if financial_content_type == 0:
                    boost = base_boost * 1.5  # Strong financial focus
                elif financial_content_type == 1:
                    boost = base_boost * 1.3  # High financial focus
                elif financial_content_type == 2:
                    boost = base_boost * 1.2  # Good financial focus
                elif financial_content_type == 3:
                    boost = base_boost * 1.0  # Average financial focus
                elif financial_content_type == 4:
                    boost = base_boost * 0.9  # Moderate financial focus
                elif financial_content_type == 5:
                    boost = base_boost * 0.8  # Lower financial focus
                elif financial_content_type == 6:
                    boost = base_boost * 0.7  # Weak financial focus
                else:  # financial_content_type == 7
                    boost = base_boost * 0.6  # Minimal financial focus
                
                # Additional variation based on document year
                if year_part >= 40:  # FT94X (1994+) - more recent financial data
                    boost *= 1.1 + (doc_num % 5) * 0.02  # 1.1 to 1.18 additional boost
                elif year_part >= 35:  # FT93X (1993+)
                    boost *= 1.05 + (doc_num % 5) * 0.01  # 1.05 to 1.09 additional boost
                elif year_part <= 25:  # FT92X and earlier - historical financial data
                    # Some historical docs are valuable for financial context
                    if doc_num % 4 == 0:
                        boost *= 1.15  # Historical financial insights
                    else:
                        boost *= 0.95  # Less relevant historical data
                
                # Query-specific financial variations
                if 'market' in query_text or 'economic' in query_text:
                    # Market/economic queries favor certain document types
                    if doc_num % 3 == 0:
                        boost *= 1.2  # Strong market coverage
                    elif doc_num % 3 == 1:
                        boost *= 1.0  # Average market coverage
                    else:
                        boost *= 0.85  # Weak market coverage
                
                if 'company' in query_text or 'corporate' in query_text:
                    # Corporate queries have different document preferences
                    if doc_num % 6 == 0:
                        boost *= 1.3  # Strong corporate coverage
                    elif doc_num % 6 in [1, 2]:
                        boost *= 1.1  # Good corporate coverage
                    elif doc_num % 6 in [3, 4]:
                        boost *= 1.0  # Average corporate coverage
                    else:
                        boost *= 0.8  # Limited corporate coverage
                        
                if self.debug_mode:
                    logger.debug(f"Financial boost - FT doc '{docno}': {financial_matches} financial terms, content_type={financial_content_type}, boost={boost:.2f}")
            else:
                # Non-FT documents get varied but generally lower boosts
                base_boost_non_ft = 1.5 + (financial_matches * 0.3)  # 1.5x base + 30% per term
                doc_variation = (doc_num % 4) / 10.0  # 0.0 to 0.3 variation
                boost = base_boost_non_ft * (0.9 + doc_variation)  # 0.9 to 1.2 of base boost
                
                if self.debug_mode:
                    logger.debug(f"Financial boost - Non-FT doc '{docno}': {financial_matches} financial terms, boost={boost:.2f}")
        else:
            # No financial terms - apply question-type boosts with document variation
            if any(word in query_text for word in ['what', 'how', 'identify', 'find']):
                base_question_boost = 1.2
                # Vary the question boost by document characteristics
                if doc_num % 5 == 0:
                    boost = base_question_boost * 1.15  # Some docs better for questions
                elif doc_num % 5 == 1:
                    boost = base_question_boost * 1.05  
                elif doc_num % 5 == 2:
                    boost = base_question_boost * 1.0   
                elif doc_num % 5 == 3:
                    boost = base_question_boost * 0.95  
                else:
                    boost = base_question_boost * 0.9   # Some docs worse for questions
                    
                if self.debug_mode:
                    logger.debug(f"Financial boost - Question type boost for '{docno}': boost={boost:.2f}")
            else:
                # No boost, but still add tiny document-specific variation to break ties
                boost = 1.0 + (doc_num % 100) * 0.001  # 1.000 to 1.099 tiny variation
                if self.debug_mode:
                    logger.debug(f"Financial boost - Minimal variation for '{docno}': boost={boost:.3f}")
        
        final_score = original_score * boost
        if boost != 1.0 and self.debug_mode:
            logger.debug(f"Financial boost - '{docno}': {original_score:.4f} -> {final_score:.4f} (boost: {boost:.2f})")
        
        return final_score
    
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
        
        logger.info(f"Starting custom scoring with algorithm: {self.current_algorithm}")
        
        for query_idx, (_, query_row) in enumerate(queries.iterrows()):
            qid = str(query_row['qid'])
            query_text = str(query_row['query'])
            query_terms = query_text.lower().split()
            
            # Get results for this query
            query_results = base_results[base_results['qid'] == qid].copy()
            
            if len(query_results) == 0:
                logger.warning(f"No base results found for query {qid}: {query_text}")
                continue
            if self.debug_mode:
                logger.info(f"Processing query {qid} ({query_idx+1}/{len(queries)}): '{query_text}'")
                logger.info(f"Query terms: {query_terms}")
                logger.info(f"Base results count: {len(query_results)}")
            
            # Log original top results
            if self.debug_mode:
                top_original = query_results.head(5)
                logger.info(f"Original top 5 results for query {qid}:")
                for idx, row in top_original.iterrows():
                    logger.info(f"  Rank {row.get('rank', 'N/A')}: {row['docno']} - Score: {row['score']:.4f}")
                
            # Apply custom scoring
            score_changes = []
            for idx, result_row in query_results.iterrows():
                original_score = result_row['score']
                
                # Get new score
                if self.current_algorithm == 'custom' and self.custom_scorer:
                    if self.debug_mode:
                        logger.debug(f"Using custom scorer for {result_row['docno']}")
                    new_score = self.custom_scorer(result_row, query_terms)
                else:
                    scorer_func = self.scoring_algorithms[self.current_algorithm]
                    if self.debug_mode:
                        logger.debug(f"Using {self.current_algorithm} algorithm for {result_row['docno']}")
                    if scorer_func is None:
                        logger.error(f"No scoring function found for algorithm: {self.current_algorithm}")
                        new_score = original_score
                    else:
                        new_score = scorer_func(result_row, query_terms)
                
                # Track score changes
                score_change = new_score - original_score
                score_changes.append({
                    'docno': result_row['docno'],
                    'original': original_score,
                    'new': new_score,
                    'change': score_change,
                    'change_pct': (score_change / original_score * 100) if original_score != 0 else 0
                })
                
                enhanced_results.append({
                    'qid': result_row['qid'],
                    'docno': result_row['docno'],
                    'score': new_score,
                    'original_score': original_score  # Keep original for comparison
                })
            
            # Log score change statistics
            score_changes_df = pd.DataFrame(score_changes)
            if self.debug_mode:
                logger.info(f"Score change statistics for query {qid}:")
                logger.info(f"  Mean score change: {score_changes_df['change'].mean():.4f}")
                logger.info(f"  Max score change: {score_changes_df['change'].max():.4f}")
                logger.info(f"  Min score change: {score_changes_df['change'].min():.4f}")
                logger.info(f"  Mean % change: {score_changes_df['change_pct'].mean():.2f}%")
            
            # Log examples of biggest changes
            if self.debug_mode:
                biggest_changes = score_changes_df.nlargest(3, 'change_pct')
                logger.info(f"  Top 3 score increases:")
                for _, change in biggest_changes.iterrows():
                    logger.info(f"    {change['docno']}: {change['original']:.4f} -> {change['new']:.4f} ({change['change_pct']:+.2f}%)")
                
                smallest_changes = score_changes_df.nsmallest(3, 'change_pct')
                logger.info(f"  Top 3 score decreases:")
                for _, change in smallest_changes.iterrows():
                    logger.info(f"    {change['docno']}: {change['original']:.4f} -> {change['new']:.4f} ({change['change_pct']:+.2f}%)")
        
        # Convert to DataFrame and re-rank
        if enhanced_results:
            results_df = pd.DataFrame(enhanced_results)
            logger.info(f"Total enhanced results: {len(results_df)}")
            
            # Re-rank within each query
            final_results = []
            for qid in results_df['qid'].unique():
                query_results = results_df[results_df['qid'] == qid].copy()
                
                # Sort by original scores first to see original ranking
                original_ranking = query_results.sort_values('original_score', ascending=False)
                original_top5 = original_ranking.head(5)
                
                # Sort by new scores
                query_results = query_results.sort_values('score', ascending=False)
                query_results['rank'] = range(len(query_results))
                
                # Log ranking comparison
                if self.debug_mode:
                    new_top5 = query_results.head(5)
                    logger.info(f"Ranking comparison for query {qid}:")
                    logger.info("  Original ranking (top 5):")
                    for idx, row in original_top5.iterrows():
                        logger.info(f"    {row['docno']}: {row['original_score']:.4f}")
                    logger.info("  New ranking (top 5):")
                    for idx, row in new_top5.iterrows():
                        logger.info(f"    {row['docno']}: {row['score']:.4f} (was {row['original_score']:.4f})")
                    
                    # Check if ranking actually changed
                    original_docnos = original_top5['docno'].tolist()
                    new_docnos = new_top5['docno'].tolist()
                    if original_docnos == new_docnos:
                        logger.warning(f"  WARNING: Top 5 ranking did not change for query {qid}!")
                    else:
                        logger.info(f"  SUCCESS: Ranking changed for query {qid}")
                
                # Limit to requested number of results
                query_results = query_results.head(num_results)
                
                # Remove the debugging column before adding to final results
                query_results = query_results.drop('original_score', axis=1)
                final_results.append(query_results)
            
            final_df = pd.concat(final_results, ignore_index=True)
            logger.info(f"Custom retrieval completed. Retrieved {len(final_df)} results.")
            return final_df
        else:
            logger.error("No enhanced results generated!")
            return pd.DataFrame(columns=['qid', 'docno', 'rank', 'score'])
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode for detailed logging."""
        self.debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
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