import pyterrier as pt
from pyterrier import IterDictIndexer
from typing import Optional, Iterator, Dict, List, Set
import os
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
from logger import get_logger

logger = get_logger("indexer")

class Indexer:
    def __init__(self, index_path: str = './indices/ft_index'):
        """Initialize the indexer with a specified index path."""
        self.index_path = index_path
        self.index_ref = None
        self.chunk_size = 1024 * 1024  # 1MB chunks
        self.stopwords: Optional[Set[str]] = None
        
        # Try to load existing index
        if os.path.exists(os.path.join(index_path, 'data.properties')):
            try:
                self.index_ref = pt.IndexFactory.of(index_path)
                logger.info(f"Successfully loaded existing index from {index_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")
                self.index_ref = None
        
    def load_stopwords(self, stopword_path: str) -> Set[str]:
        """
        Load stopwords from a file.
        
        Args:
            stopword_path (str): Path to the stopword file
            
        Returns:
            Set of stopwords
        """
        stopwords = set()
        try:
            with open(stopword_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:  # Skip empty lines
                        stopwords.add(word)
            logger.info(f"Loaded {len(stopwords)} stopwords from {stopword_path}")
            self.stopwords = stopwords
            return stopwords
        except Exception as e:
            logger.error(f"Error loading stopwords from {stopword_path}: {str(e)}")
            return set()
        
    def parse_sgml_chunk(self, chunk: str) -> List[Dict]:
        """
        Parse a chunk of SGML text and return a list of document dictionaries.
        Each document is expected to be enclosed in <DOC>...</DOC> tags.
        """
        docs = []
        doc_pattern = re.compile(r'<DOC>(.*?)</DOC>', re.DOTALL)
        for doc_match in doc_pattern.finditer(chunk):
            doc_text = doc_match.group(1)
            doc_dict = {}
            
            # Extract DOCNO
            docno_match = re.search(r'<DOCNO>(.*?)</DOCNO>', doc_text, re.DOTALL)
            if docno_match:
                doc_dict['docno'] = docno_match.group(1).strip()
            
            # Extract HEADLINE
            headline_match = re.search(r'<HEADLINE>(.*?)</HEADLINE>', doc_text, re.DOTALL)
            if headline_match:
                doc_dict['title'] = headline_match.group(1).strip()
            else:
                doc_dict['title'] = ''  # Empty string for missing titles
            
            # Extract TEXT
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc_text, re.DOTALL)
            if text_match:
                doc_dict['body'] = text_match.group(1).strip()
            else:
                doc_dict['body'] = ''  # Empty string for missing body text
            
            if doc_dict.get('docno'):  # Only add if we have at least a docno
                docs.append(doc_dict)
        return docs

    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single file in chunks and return all documents."""
        all_docs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    all_docs.extend(self.parse_sgml_chunk(chunk))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
        return all_docs

    def process_files_parallel(self, file_paths: List[str], max_workers: Optional[int] = None) -> Iterator[Dict]:
        """Process multiple files in parallel."""
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_file, file_path) for file_path in file_paths]
            for future in tqdm(futures, desc="Processing files"):
                docs = future.result()
                for doc in docs:
                    yield doc

    def index_corpus(self, corpus_path: str, fields: list = ['title', 'body'], stopword_path: Optional[str] = None) -> None:
        """
        Index the corpus with specified fields using parallel processing.
        
        Args:
            corpus_path (str): Path to the corpus directory
            fields (list): List of fields to index
            stopword_path (str, optional): Path to stopword file
        """
        # Load stopwords if provided
        if stopword_path and os.path.exists(stopword_path):
            self.load_stopwords(stopword_path)
        
        # Get all .txt files in the directory
        file_paths = [
            os.path.join(corpus_path, f) 
            for f in os.listdir(corpus_path) 
            if f.endswith('.txt')
        ]
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Configure indexer properties
        properties = {
            'index.document.class': 'FSAFieldDocumentIndex',
            'indexer.meta.forward.keys': 'docno',
            'indexer.meta.reverse.keys': 'docno',
            'indexer.meta.forward.keylens': '20',
            'indexer.meta.reverse.keylens': '20'
        }
        
        # Add stopword configuration if stopwords are loaded
        if self.stopwords:
            # Create a stopword file in the index directory
            stopword_file = os.path.join(self.index_path, 'stopwords.lst')
            with open(stopword_file, 'w', encoding='utf-8') as f:
                for word in sorted(self.stopwords):
                    f.write(f"{word}\n")
            
            properties.update({
                'stopwords.filename': stopword_file,
                'termpipelines': 'Stopwords,PorterStemmer'
            })
            logger.info(f"Configured indexing with {len(self.stopwords)} stopwords and stemming")
        else:
            properties['termpipelines'] = 'PorterStemmer'
            logger.info("Configured indexing with stemming only (no stopwords)")
        
        # Initialize the indexer with field-based indexing
        indexer = IterDictIndexer(
            self.index_path,
            fields=fields,
            text_attrs=fields,
            meta=['docno'],
            properties=properties
        )
        
        # Process files in parallel and index documents as they come in
        self.index_ref = indexer.index(self.process_files_parallel(file_paths))
        
        logger.info("Indexing completed successfully")
        
    def get_index_ref(self):
        """Return the index reference."""
        if self.index_ref is None:
            raise ValueError("Index has not been created yet. Call index_corpus first.")
        return self.index_ref 