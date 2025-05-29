"""
Retrieval Playground - Experiment with custom retrieval algorithms

This script demonstrates how to use the CustomRetriever class to implement
and test different scoring algorithms for information retrieval.
"""

import argparse
import math
import numpy as np
from indexer import Indexer
from query_processor import QueryProcessor
from custom_retriever import CustomRetriever
from evaluator import Evaluator
from trec_data import TRECDataLoader
from logger import get_logger

logger = get_logger("playground")

def custom_scoring_example_1(tf, df, doc_length, collection_stats, query_terms):
    """
    Example custom scoring function: BM25 with query-dependent parameters
    """
    # Adjust k1 based on query length
    query_length = len(query_terms)
    k1 = 1.2 + (query_length - 3) * 0.1  # Increase k1 for longer queries
    b = 0.75
    
    N = collection_stats['num_docs']
    avgdl = collection_stats['avg_doc_length']
    
    if N == 0 or df == 0:
        return 0.0
    
    idf = math.log((N - df + 0.5) / (df + 0.5))
    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))
    
    return idf * tf_component

def custom_scoring_example_2(tf, df, doc_length, collection_stats, query_terms):
    """
    Example custom scoring function: Combination of BM25 and TF-IDF
    """
    N = collection_stats['num_docs']
    avgdl = collection_stats['avg_doc_length']
    
    if N == 0 or df == 0:
        return 0.0
    
    # BM25 component
    k1, b = 1.2, 0.75
    bm25_idf = math.log((N - df + 0.5) / (df + 0.5))
    bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))
    bm25_score = bm25_idf * bm25_tf
    
    # TF-IDF component
    tfidf_tf = 1 + math.log(tf) if tf > 0 else 0
    tfidf_idf = math.log(N / df)
    tfidf_score = tfidf_tf * tfidf_idf
    
    # Weighted combination
    alpha = 0.7  # Weight for BM25
    return alpha * bm25_score + (1 - alpha) * tfidf_score

def custom_scoring_example_3(tf, df, doc_length, collection_stats, query_terms):
    """
    Example custom scoring function: Document length penalty
    """
    N = collection_stats['num_docs']
    avgdl = collection_stats['avg_doc_length']
    
    if N == 0 or df == 0:
        return 0.0
    
    # Standard BM25
    k1, b = 1.2, 0.75
    idf = math.log((N - df + 0.5) / (df + 0.5))
    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))
    bm25_score = idf * tf_component
    
    # Add document length penalty for very long or very short documents
    length_ratio = doc_length / avgdl
    if length_ratio > 2.0:  # Very long documents
        penalty = 0.9
    elif length_ratio < 0.5:  # Very short documents
        penalty = 0.8
    else:
        penalty = 1.0
    
    return bm25_score * penalty

def compare_algorithms(index_ref, queries, qrels, k=10):
    """Compare different retrieval algorithms."""
    logger.info("=== ALGORITHM COMPARISON ===")
    
    algorithms = ['tf_idf', 'bm25', 'bm25_plus', 'cosine', 'language_model']
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"\n--- Testing {algorithm.upper()} ---")
        
        # Initialize custom retriever
        retriever = CustomRetriever(index_ref, verbose=False)
        retriever.set_scoring_algorithm(algorithm)
        
        # Retrieve documents
        retrieved = retriever.retrieve(queries, num_results=1000)
        
        # Evaluate
        evaluator = Evaluator(k=k)
        scores = evaluator.evaluate(retrieved, qrels)
        
        results[algorithm] = scores
        logger.info(f"{algorithm}: NDCG@{k}={scores['ndcg']:.4f}, MAP@{k}={scores['map']:.4f}")
    
    return results

def test_custom_scorers(index_ref, queries, qrels, k=10):
    """Test custom scoring functions."""
    logger.info("\n=== CUSTOM SCORING FUNCTIONS ===")
    
    custom_functions = [
        ("Query-dependent BM25", custom_scoring_example_1),
        ("BM25 + TF-IDF Hybrid", custom_scoring_example_2),
        ("BM25 with Length Penalty", custom_scoring_example_3)
    ]
    
    results = {}
    
    for name, func in custom_functions:
        logger.info(f"\n--- Testing {name} ---")
        
        # Initialize custom retriever
        retriever = CustomRetriever(index_ref, verbose=False)
        retriever.set_custom_scorer(func)
        
        # Retrieve documents
        retrieved = retriever.retrieve(queries, num_results=1000)
        
        # Evaluate
        evaluator = Evaluator(k=k)
        scores = evaluator.evaluate(retrieved, qrels)
        
        results[name] = scores
        logger.info(f"{name}: NDCG@{k}={scores['ndcg']:.4f}, MAP@{k}={scores['map']:.4f}")
    
    return results

def parameter_tuning_example(index_ref, queries, qrels, k=10):
    """Example of parameter tuning for BM25."""
    logger.info("\n=== BM25 PARAMETER TUNING ===")
    
    # Test different k1 and b values
    k1_values = [0.8, 1.0, 1.2, 1.5, 2.0]
    b_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    best_score = 0
    best_params = {}
    
    for k1 in k1_values:
        for b in b_values:
            retriever = CustomRetriever(index_ref, verbose=False)
            retriever.set_scoring_algorithm('bm25', k1=k1, b=b)
            
            retrieved = retriever.retrieve(queries, num_results=1000)
            
            evaluator = Evaluator(k=k)
            scores = evaluator.evaluate(retrieved, qrels)
            
            ndcg = scores['ndcg']
            logger.info(f"k1={k1}, b={b}: NDCG@{k}={ndcg:.4f}")
            
            if ndcg > best_score:
                best_score = ndcg
                best_params = {'k1': k1, 'b': b}
    
    logger.info(f"\nBest parameters: {best_params} with NDCG@{k}={best_score:.4f}")
    return best_params

def main():
    parser = argparse.ArgumentParser(description='Retrieval Algorithm Playground')
    parser.add_argument('--index-path', type=str, default='./indices/ft_index',
                       help='Path to the index')
    parser.add_argument('--trec-data-path', type=str, default='data/query-relJudgments',
                       help='Path to TREC data')
    parser.add_argument('--k', type=int, default=10,
                       help='Evaluation cutoff')
    parser.add_argument('--experiment', type=str, choices=['compare', 'custom', 'tuning', 'all'], 
                       default='all', help='Which experiment to run')
    
    args = parser.parse_args()
    
    # Initialize PyTerrier
    import pyterrier as pt
    pt.java.init()
    
    # Load index
    indexer = Indexer(args.index_path)
    index_ref = indexer.get_index_ref()
    
    # Load TREC data
    trec_loader = TRECDataLoader(args.trec_data_path)
    queries = trec_loader.get_queries()
    qrels = trec_loader.get_qrels(filter_collections=['FT'])
    
    # Filter to queries with qrels
    queries_with_qrels = trec_loader.get_queries_with_qrels(filter_collections=['FT'])
    queries = queries_with_qrels
    
    logger.info(f"Loaded {len(queries)} queries with relevance judgments")
    
    # Run experiments
    if args.experiment in ['compare', 'all']:
        compare_algorithms(index_ref, queries, qrels, args.k)
    
    if args.experiment in ['custom', 'all']:
        test_custom_scorers(index_ref, queries, qrels, args.k)
    
    if args.experiment in ['tuning', 'all']:
        parameter_tuning_example(index_ref, queries, qrels, args.k)

if __name__ == "__main__":
    main() 