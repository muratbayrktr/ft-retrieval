import pyterrier as pt
from indexer import Indexer
from query_processor import QueryProcessor
from retriever import Retriever
from evaluator import Evaluator
from trec_data import TRECDataLoader
from logger import get_logger
import os
import shutil
import argparse
import json

# Get logger instance
logger = get_logger("main")

def ensure_directory(directory):
    """Ensure that a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def clean_index_directory(directory):
    """Clean the index directory if it exists."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logger.info(f"Cleaned existing index directory: {directory}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TREC FT Retrieval System')
    parser.add_argument('--rebuild-index', action='store_true',
                      help='Force rebuild the index even if it exists')
    parser.add_argument('--corpus-path', type=str, default='data/ft/all',
                      help='Path to the corpus directory (default: data/ft/all)')
    parser.add_argument('--index-path', type=str, default='./indices/ft_index',
                      help='Path to store the index (default: ./indices/ft_index)')
    parser.add_argument('--trec-data-path', type=str, default='data/query-relJudgments',
                      help='Path to TREC data directory (default: data/query-relJudgments)')
    parser.add_argument('--k', type=int, default=20,
                      help='Number of top results to consider for evaluation (default: 20)')
    parser.add_argument('--use-stopwords', action='store_true', default=True,
                      help='Use stopwords during indexing (default: True)')
    parser.add_argument('--no-stopwords', action='store_true',
                      help='Disable stopword usage during indexing')
    parser.add_argument('--retriever', type=str, choices=['pyterrier', 'custom'], 
                      default='pyterrier', help='Which retriever to use (default: pyterrier)')
    parser.add_argument('--algorithm', type=str, 
                      choices=['tf_idf', 'bm25', 'bm25_plus', 'cosine', 'language_model', 'financial_boost', 'reverse_test', 'random_test'],
                      default='bm25', help='Custom retriever algorithm (default: bm25)')
    parser.add_argument('--save-results', action='store_true',
                      help='Save results to a file')
    return parser.parse_args()

def analyze_results(results, qrels, queries):
    """Analyze retrieval results and qrels to identify potential issues."""
    logger.info("\n=== Analysis of Results ===")
    
    # Sample queries
    logger.info("\nSample Queries:")
    for i in range(min(5, len(queries))):
        logger.info(f"Query {queries.iloc[i]['qid']}: {queries.iloc[i]['query'][:100]}...")
    
    # Sample results
    logger.info("\nSample Retrieved Results:")
    sample_qids = results['qid'].unique()[:3]
    for qid in sample_qids:
        qid_results = results[results['qid'] == qid].head(5)
        logger.info(f"\nQuery {qid} top 5 results:")
        for _, row in qid_results.iterrows():
            logger.info(f"  - Doc: {row['docno']}, Score: {row['score']:.4f}")
    
    # Sample qrels
    logger.info("\nSample Relevance Judgments:")
    qrels_qids = qrels['qid'].unique()
    logger.info(f"Total unique queries in qrels: {len(qrels_qids)}")
    logger.info(f"Sample qids from qrels: {list(qrels_qids[:10])}")
    
    # Check for relevant documents
    relevant_qrels = qrels[qrels['label'] > 0]
    logger.info(f"\nTotal relevant judgments: {len(relevant_qrels)}")
    logger.info("\nSample relevant documents:")
    for i in range(min(10, len(relevant_qrels))):
        row = relevant_qrels.iloc[i]
        logger.info(f"  Query {row['qid']}: Doc {row['docno']} (relevance: {row['label']})")
    
    # Check overlap between results and qrels
    results_qids = set(results['qid'].unique())
    qrels_qids = set(qrels['qid'].unique())
    overlap_qids = results_qids.intersection(qrels_qids)
    logger.info(f"\nQuery ID overlap: {len(overlap_qids)} queries")
    logger.info(f"Queries in results but not in qrels: {results_qids - qrels_qids}")
    logger.info(f"Queries in qrels but not in results: {qrels_qids - results_qids}")
    
    # Check document overlap for overlapping queries
    if overlap_qids:
        sample_qid = list(overlap_qids)[0]
        results_docs = set(results[results['qid'] == sample_qid]['docno'])
        qrels_docs = set(qrels[qrels['qid'] == sample_qid]['docno'])
        overlap_docs = results_docs.intersection(qrels_docs)
        logger.info(f"\nFor query {sample_qid}:")
        logger.info(f"  Retrieved docs: {len(results_docs)}")
        logger.info(f"  Judged docs: {len(qrels_docs)}")
        logger.info(f"  Overlap: {len(overlap_docs)}")
        if overlap_docs:
            logger.info(f"  Sample overlapping docs: {list(overlap_docs)[:5]}")

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize PyTerrier
        pt.java.init()  # Force Java initialization
        logger.info("PyTerrier initialized successfully")
        
        # Handle index directory
        if args.rebuild_index:
            logger.info("Rebuilding index as requested")
            clean_index_directory(args.index_path)
        ensure_directory(args.index_path)
        
        # Initialize indexer
        indexer = Indexer(args.index_path)
        
        # Check if index exists
        index_exists = os.path.exists(os.path.join(args.index_path, 'data.properties'))
        
        # Index corpus if needed
        if not index_exists or args.rebuild_index:
            logger.info("Starting indexing process...")
            if not os.path.exists(args.corpus_path):
                raise FileNotFoundError(f"Corpus directory not found: {args.corpus_path}")
            
            # Use stopword file if it exists and user wants to use stopwords
            use_stopwords = args.use_stopwords and not args.no_stopwords
            stopword_path = os.path.join(args.corpus_path, 'stopword.lst')
            
            if use_stopwords and os.path.exists(stopword_path):
                logger.info(f"Found stopword file at {stopword_path}, using stopwords for indexing")
                indexer.index_corpus(args.corpus_path, fields=['title', 'body'], stopword_path=stopword_path)
            else:
                if not use_stopwords:
                    logger.info("Stopword usage disabled by user")
                elif not os.path.exists(stopword_path):
                    logger.info(f"No stopword file found at {stopword_path}")
                logger.info("Indexing without stopwords")
                indexer.index_corpus(args.corpus_path, fields=['title', 'body'])
            logger.info("Indexing completed successfully")
        else:
            logger.info("Using existing index")
        
        # Load TREC data
        logger.info("Loading TREC data...")
        trec_loader = TRECDataLoader(args.trec_data_path)
        queries = trec_loader.get_queries()
        
        # Filter qrels to only include documents from collections we have indexed (FT)
        qrels = trec_loader.get_qrels(filter_collections=['FT'])
        
        # Show dataset statistics
        stats = trec_loader.get_dataset_stats(filter_collections=['FT'])
        logger.info(f"Dataset stats (FT collection only): \n{json.dumps(stats, indent=4)}")
        
        # Filter queries to only those with relevance judgments
        if qrels is not None and len(qrels) > 0:
            queries_with_qrels = trec_loader.get_queries_with_qrels(filter_collections=['FT'])
            logger.info(f"Filtered queries from {len(queries)} to {len(queries_with_qrels)} (those with FT relevance judgments)")
            queries = queries_with_qrels
        else:
            logger.warning("No relevance judgments found for FT collection")
            qrels = None
        
        # 2. Query Processing
        logger.info("Initializing query processor...")
        query_processor = QueryProcessor()
        query_processor.load_queries(queries)
        
        # 3. Retrieval
        logger.info("Initializing retriever...")
        
        if args.retriever == 'custom':
            from custom_retriever import CustomRetriever
            retriever = CustomRetriever(indexer.get_index_ref(), verbose=True, debug_mode=False)
            retriever.set_scoring_algorithm(args.algorithm)
            logger.info(f"Using custom retriever with {args.algorithm} algorithm")
            results = retriever.retrieve(query_processor.get_queries(), num_results=1000)
        else:
            retriever = Retriever(indexer.get_index_ref())
            results = retriever.retrieve(query_processor.get_queries())
            logger.info("Using PyTerrier BM25 retriever")
        
        logger.info("Retrieval completed successfully")
        
        # Analyze results before evaluation
        # analyze_results(results, qrels, queries)
        
        # 4. Evaluation
        logger.info("Initializing evaluator...")
        evaluator = Evaluator(k=args.k)
        if qrels is not None:
            scores = evaluator.evaluate(results, qrels)
            if args.save_results:
                ensure_directory('./results')
                experiment_name = f'{args.retriever}_{args.algorithm}_k{args.k}'
                if args.rebuild_index:
                    experiment_name += f'_rebuild_index_{"with" if args.use_stopwords else "without"}_stopwords'
                with open(f'./results/{experiment_name}.json', 'w+') as f:
                    json.dump(scores, f, indent=4)
        else:
            logger.warning("No relevance judgments available, skipping evaluation")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()