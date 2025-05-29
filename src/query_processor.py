import pandas as pd
from typing import List, Dict
from logger import get_logger
import re

logger = get_logger("query_processor")

class QueryProcessor:
    def __init__(self):
        """Initialize the query processor."""
        self.queries = None
        
    def preprocess_query(self, query_text: str) -> str:
        """
        Preprocess a query text to make it suitable for PyTerrier.
        
        Args:
            query_text (str): Original query text
            
        Returns:
            str: Preprocessed query text
        """
        # Remove newlines and extra whitespace
        query = ' '.join(query_text.split())
        
        # Replace special characters that might cause parsing issues
        # These characters have special meaning in PyTerrier's query parser
        special_chars = {
            '?': '',      # Remove question marks
            '!': '',      # Remove exclamation marks
            '(': ' ',     # Replace parentheses with spaces
            ')': ' ',
            '/': ' ',     # Replace forward slash with space (e.g., and/or -> and or)
            '\\': ' ',    # Replace backslash with space
            '+': ' ',     # Replace plus with space
            '-': ' ',     # Replace hyphen/minus with space
            '^': ' ',     # Replace caret with space
            ':': ' ',     # Replace colon with space
            '~': ' ',     # Replace tilde with space
            '"': ' ',     # Replace quotes with space
            "'": ' ',     # Replace single quotes with space
            '[': ' ',     # Replace brackets with space
            ']': ' ',
            '{': ' ',     # Replace braces with space
            '}': ' ',
            '|': ' ',     # Replace pipe with space
            '&': ' ',     # Replace ampersand with space
            '*': ' ',     # Replace asterisk with space
            '%': ' ',     # Replace percent with space
            '#': ' ',     # Replace hash with space
            '@': ' ',     # Replace at symbol with space
            '$': ' ',     # Replace dollar sign with space
            '=': ' ',     # Replace equals with space
            '<': ' ',     # Replace less than with space
            '>': ' ',     # Replace greater than with space
            ';': ' ',     # Replace semicolon with space
            ',': ' ',     # Replace comma with space
        }
        
        # Apply all replacements
        for char, replacement in special_chars.items():
            query = query.replace(char, replacement)
        
        # Remove any multiple spaces and trim
        query = ' '.join(query.split())
        
        return query
        
    def load_queries(self, queries_df: pd.DataFrame) -> None:
        """
        Load and preprocess queries from a DataFrame.
        
        Args:
            queries_df (pd.DataFrame): DataFrame containing queries with 'qid' and 'query' columns
        """
        if not isinstance(queries_df, pd.DataFrame):
            raise ValueError("queries_df must be a pandas DataFrame")
            
        if 'qid' not in queries_df.columns or 'query' not in queries_df.columns:
            raise ValueError("queries_df must contain 'qid' and 'query' columns")
            
        # Create a copy of the DataFrame
        self.queries = queries_df.copy()
        
        # Preprocess each query
        self.queries['query'] = self.queries['query'].apply(self.preprocess_query)
        
        logger.info(f"Loaded and preprocessed {len(self.queries)} queries")
        
    def get_queries(self) -> pd.DataFrame:
        """
        Get the preprocessed queries.
        
        Returns:
            pd.DataFrame: DataFrame containing preprocessed queries
        """
        if self.queries is None:
            raise ValueError("No queries loaded. Call load_queries first.")
        return self.queries 