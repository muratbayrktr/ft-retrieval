import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from logger import get_logger

logger = get_logger("trec_data")

class TRECDataLoader:
    def __init__(self, data_dir: str = 'data/query-relJudgments'):
        """
        Initialize the TREC data loader.
        
        Args:
            data_dir (str): Path to the directory containing TREC data files
        """
        self.data_dir = Path(data_dir)
        self.topics = {}
        self.qrels = None
        self._cached_queries = None
        self._cached_qrels = None
        
    def parse_topic_file(self, file_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Parse a TREC topic file and extract queries.
        
        Args:
            file_path (Path): Path to the topic file
            
        Returns:
            Dict mapping query IDs to query information
        """
        topics = {}
        current_topic = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split into individual topics
            topic_blocks = re.split(r'<top>', content)[1:]  # Skip first empty split
            
            for block in topic_blocks:
                # Extract query number
                num_match = re.search(r'<num>\s*Number:\s*(\d+)', block)
                if not num_match:
                    logger.warning(f"Could not find query number in block: {block[:100]}...")
                    continue
                qid = num_match.group(1)
                
                # Extract title
                title_match = re.search(r'<title>\s*(.*?)(?=<desc>|<narr>|</top>)', block, re.DOTALL)
                title = title_match.group(1).strip() if title_match else ""
                
                # Extract description
                desc_match = re.search(r'<desc>\s*Description:\s*(.*?)(?=<narr>|</top>)', block, re.DOTALL)
                desc = desc_match.group(1).strip() if desc_match else ""
                
                # Extract narrative
                narr_match = re.search(r'<narr>\s*Narrative:\s*(.*?)(?=</top>)', block, re.DOTALL)
                narr = narr_match.group(1).strip() if narr_match else ""
                
                topics[qid] = {
                    'title': title,
                    'description': desc,
                    'narrative': narr
                }
                
            return topics
        except Exception as e:
            logger.error(f"Error parsing topic file {file_path}: {str(e)}")
            return {}
    
    def load_topics(self) -> None:
        """Load all topic files from the data directory."""
        topic_files = list(self.data_dir.glob('q-topics-org-SET*.txt'))
        if not topic_files:
            logger.warning(f"No topic files found in {self.data_dir}")
            return
            
        for file_path in topic_files:
            logger.info(f"Loading topics from {file_path.name}")
            topics = self.parse_topic_file(file_path)
            self.topics.update(topics)
        logger.info(f"Loaded {len(self.topics)} topics")
    
    def load_qrels(self) -> None:
        """Load all qrels files from the data directory."""
        qrels_files = list(self.data_dir.glob('qrels*.txt'))
        if not qrels_files:
            logger.warning(f"No qrels files found in {self.data_dir}")
            return
            
        qrels_data = []
        
        for file_path in qrels_files:
            logger.info(f"Loading qrels from {file_path.name}")
            try:
                # Read qrels file
                df = pd.read_csv(file_path, sep='\s+', header=None,
                               names=['qid', 'iter', 'docno', 'rel'])
                # Convert qid to string to match topics DataFrame
                df['qid'] = df['qid'].astype(str)
                qrels_data.append(df)
            except Exception as e:
                logger.error(f"Error loading qrels file {file_path}: {str(e)}")
                continue
        
        if qrels_data:
            self.qrels = pd.concat(qrels_data, ignore_index=True)
            logger.info(f"Loaded {len(self.qrels)} relevance judgments")
        else:
            logger.warning("No valid qrels data found")
            
    def filter_qrels_by_collection(self, collection_prefixes: List[str]) -> pd.DataFrame:
        """
        Filter qrels to only include documents from specified collections.
        
        Args:
            collection_prefixes (List[str]): List of collection prefixes to keep (e.g., ['FT'])
            
        Returns:
            Filtered qrels DataFrame
        """
        if self.qrels is None:
            self.load_qrels()
            
        if self.qrels is None:
            return pd.DataFrame()
            
        # Create a pattern to match any of the collection prefixes
        pattern = '|'.join([f'^{prefix}' for prefix in collection_prefixes])
        
        # Filter documents that match the pattern
        filtered_qrels = self.qrels[self.qrels['docno'].str.match(pattern, na=False)]
        
        # Only keep queries that still have relevance judgments after filtering
        queries_with_judgments = filtered_qrels['qid'].unique()
        filtered_qrels = filtered_qrels[filtered_qrels['qid'].isin(queries_with_judgments)]
        
        logger.info(f"Filtered qrels from {len(self.qrels)} to {len(filtered_qrels)} judgments")
        logger.info(f"Keeping {len(queries_with_judgments)} queries with judgments from collections: {collection_prefixes}")
        
        return filtered_qrels
    
    def get_queries(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Get queries in a format suitable for PyTerrier.
        
        Args:
            use_cache (bool): Whether to use cached queries if available
            
        Returns:
            DataFrame with columns 'qid' and 'query'
        """
        if use_cache and self._cached_queries is not None:
            return self._cached_queries
            
        if not self.topics:
            self.load_topics()
            
        queries = []
        for qid, topic in self.topics.items():
            # Combine title and description for the query
            query = f"{topic['title']} {topic['description']}"
            queries.append({'qid': qid, 'query': query})
            
        self._cached_queries = pd.DataFrame(queries)
        return self._cached_queries
    
    def get_qrels(self, use_cache: bool = True, filter_collections: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Get relevance judgments in a format suitable for PyTerrier.
        
        Args:
            use_cache (bool): Whether to use cached qrels if available
            filter_collections (List[str], optional): Collection prefixes to filter by (e.g., ['FT'])
            
        Returns:
            DataFrame with columns 'qid', 'docno', and 'label'
        """
        if use_cache and self._cached_qrels is not None and filter_collections is None:
            return self._cached_qrels
            
        if self.qrels is None:
            self.load_qrels()
            
        if self.qrels is not None:
            qrels_df = self.qrels.copy()
            
            # Apply collection filtering if specified
            if filter_collections:
                qrels_df = self.filter_qrels_by_collection(filter_collections)
            
            # Rename 'rel' to 'label' for PyTerrier compatibility
            result = qrels_df.rename(columns={'rel': 'label'})
            
            if not filter_collections:
                self._cached_qrels = result
                
            return result
        return None
        
    def get_query_components(self, qid: str) -> Dict[str, str]:
        """
        Get individual components of a query.
        
        Args:
            qid (str): Query ID
            
        Returns:
            Dict containing title, description, and narrative
        """
        if not self.topics:
            self.load_topics()
            
        return self.topics.get(qid, {})
        
    def get_queries_with_qrels(self, filter_collections: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get only queries that have relevance judgments.
        
        Args:
            filter_collections (List[str], optional): Collection prefixes to filter by
        
        Returns:
            DataFrame with columns 'qid' and 'query'
        """
        qrels = self.get_qrels(filter_collections=filter_collections)
        if qrels is None or len(qrels) == 0:
            return pd.DataFrame()
            
        queries = self.get_queries()
        qrels_qids = set(qrels['qid'].unique())
        return queries[queries['qid'].isin(qrels_qids)]
        
    def get_dataset_stats(self, filter_collections: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Get statistics about the dataset.
        
        Args:
            filter_collections (List[str], optional): Collection prefixes to filter by
            
        Returns:
            Dict containing various statistics
        """
        stats = {
            'num_topics': len(self.topics),
            'num_qrels': 0,
            'num_queries_with_qrels': 0,
            'num_relevant_docs': 0
        }
        
        qrels = self.get_qrels(filter_collections=filter_collections)
        if qrels is not None:
            stats['num_qrels'] = len(qrels)
            stats['num_queries_with_qrels'] = len(qrels['qid'].unique())
            stats['num_relevant_docs'] = len(qrels[qrels['label'] > 0])
            
        return stats 