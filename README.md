# FT Retrieval: Custom Information Retrieval Experiments

## Overview

This project implements custom information retrieval algorithms on the TREC Financial Times (FT) collection. We've built a comprehensive framework for experimenting with various retrieval approaches beyond the baseline BM25, focusing on domain-specific enhancements and novel scoring methods.

### Dataset
- **Collection**: TREC Financial Times (FT) corpus
- **Queries**: 50 test queries with relevance judgments
- **Domain**: Financial news and documents
- **Evaluation Metrics**: NDCG@10, MAP@10

## Current Implementation Status

### ✅ Completed Features
- **Baseline BM25 Implementation**: Working PyTerrier-based retrieval system
- **Custom Retrieval Framework**: Extensible system for implementing custom scoring algorithms
- **Multiple Algorithm Implementations**: 8 different retrieval approaches
- **Evaluation Pipeline**: Automated NDCG@10 and MAP@10 calculation
- **Command-line Interface**: Easy testing of individual algorithms
- **Comparison Tools**: Batch testing and result analysis

### 🔧 Technical Architecture
- **Base Retriever**: PyTerrier BM25 with custom post-processing
- **Custom Framework**: Score modification system with query-aware processing
- **Extensible Design**: Easy addition of new algorithms via function registration
- **Error Handling**: Robust fallback mechanisms for Java integration issues

## Current Results

### Baseline Performance
```
BM25 (with stopwords): NDCG@10: 0.3505, MAP@10: 0.1313
BM25 (no stopwords):   NDCG@10: 0.3116, MAP@10: 0.1165
```
**Key Finding**: Stopwords provide ~12-13% improvement over baseline

### Custom Algorithm Performance
## Retrieval Evaluation Results (k=20)

| Method                                      | NDCG   | MAP    | Precision (P) | Recall (R) |
|---------------------------------------------|--------|--------|----------------|------------|
| `custom_bm25_k20`                           | 0.3366 | 0.1533 | 0.2152         | 0.2465     |
| `custom_bm25_plus_k20`                      | 0.3479 | 0.1613 | 0.2217         | 0.2749     |
| `custom_cosine_k20`                         | 0.1692 | 0.0558 | 0.1152         | 0.1263     |
| `custom_financial_boost_k20`                | 0.3419 | 0.1616 | 0.2152         | 0.2503     |
| `custom_language_model_k20`                | 0.3142 | 0.1438 | 0.2015         | 0.2226     |
| `custom_tf_idf_k20`                         | 0.3398 | 0.1619 | 0.2152         | 0.2547     |
| `pyterrier_bm25_k20_rebuild_index_with_stopwords` | **0.3633** | **0.1719** | **0.2268**     | **0.2835**   |
| Random Test | 0.0274 | 0.0032 | - | - | -92% (verification) |

### Key Insights
- **System Verification**: Random scoring shows 92% performance drop, proving framework functionality
- **Algorithm Sensitivity**: Different approaches show measurable variations (up to 25% differences)
- **Domain Specificity**: Financial domain boosting maintains competitive performance
- **Ranking Stability**: Subtle score modifications may not change ranking order significantly

## Installation & Setup

### Prerequisites
```bash
# Python dependencies
pip install pyterrier pandas numpy ir-measures

# Java requirements (for PyTerrier)
# Ensure Java 8+ is installed
```

### Data Preparation
```bash
# Index creation (automatically handled by main.py)
python src/main.py --algorithm bm25 --query_id 301
```

## Usage

### Command Line Interface

#### Test Individual Algorithms
```bash
# Test BM25 baseline
python src/main.py --algorithm bm25 --query_id 301

# Test custom TF-IDF
python src/main.py --algorithm tf_idf --query_id 301

# Test financial domain boosting
python src/main.py --algorithm financial_boost --query_id 301
```

#### Available Algorithms
- `bm25`: Standard BM25 baseline
- `tf_idf`: Custom TF-IDF with frequency boosting
- `bm25_plus`: Enhanced BM25 with term frequency emphasis
- `cosine`: Cosine similarity-based scoring
- `language_model`: Statistical language model approach
- `financial_boost`: Domain-specific financial term enhancement
- `reverse_test`: Ranking reversal (for testing)
- `random_test`: Random scoring (for verification)

#### Batch Algorithm Comparison
```bash
# Compare multiple algorithms
python src/retrieval_playground.py
```

### Adding Custom Algorithms

Create new algorithms by adding functions to `src/custom_retriever.py`:

```python
def my_custom_algorithm(row, query_terms):
    """
    Custom scoring function
    
    Args:
        row: Document data (score, docno, etc.)
        query_terms: List of query terms
    
    Returns:
        float: Modified score
    """
    original_score = row['score']
    # Your custom logic here
    return modified_score

# Register the algorithm
CUSTOM_ALGORITHMS['my_algorithm'] = my_custom_algorithm
```

### Configuration Options

#### Stopword Handling
```bash
# With stopwords (default, better performance)
python src/main.py --algorithm bm25 --query_id 301

# Without stopwords
python src/main.py --algorithm bm25 --query_id 301 --no-stopwords
```

#### Index Management
```bash
# Force reindexing
rm -rf ./index
python src/main.py --algorithm bm25 --query_id 301
```

## Project Structure

```
ft-retrieval/
├── README.md                 # This file
├── src/
│   ├── main.py              # Command-line interface
│   ├── custom_retriever.py  # Custom algorithm implementations
│   ├── retrieval_playground.py # Algorithm comparison tool
│   └── utils.py             # Utility functions
├── data/
│   ├── ft/                  # FT collection data
│   ├── topics/              # Query topics
│   └── qrels/               # Relevance judgments
└── index/                   # PyTerrier index (auto-generated)
```

## Technical Details

### Custom Retrieval Framework

Our framework uses PyTerrier's BM25 as a base retriever and applies custom post-processing:

1. **Base Retrieval**: PyTerrier BM25 generates initial ranking
2. **Score Modification**: Custom algorithms modify document scores
3. **Re-ranking**: Documents are re-sorted by modified scores
4. **Evaluation**: Standard IR metrics calculated on final ranking

### Algorithm Design Principles

- **Conservative Modifications**: Preserve reasonable baseline performance
- **Query-Aware Processing**: Algorithms can access query terms for context
- **Extensible Architecture**: Easy addition of new scoring methods
- **Robust Fallbacks**: Graceful handling of edge cases and errors

### Known Issues & Solutions

- **Java Integration**: Some PyTerrier operations may fail; framework includes fallback mechanisms
- **Score Sensitivity**: Small score modifications may not change rankings; use aggressive multipliers for distinct results
- **Memory Usage**: Large result sets handled efficiently through pandas operations

## Future Directions

### Planned Enhancements
- **Neural Retrieval**: Integration of transformer-based models
- **Query Expansion**: Automatic query term expansion using financial lexicons
- **Learning-to-Rank**: Machine learning-based ranking optimization
- **Domain Adaptation**: Specialized financial document understanding

### Research Questions
- How do domain-specific modifications impact retrieval effectiveness?
- What is the optimal balance between precision and recall for financial documents?
- Can query analysis improve retrieval performance?

## Evaluation Methodology

### Metrics
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **MAP@10**: Mean Average Precision at rank 10

### Test Queries
- 50 TREC queries with manual relevance judgments
- Diverse financial topics and information needs
- Standard IR evaluation protocols

### Statistical Significance
- Performance differences >5% considered meaningful
- Random baseline confirms system sensitivity
- Multiple algorithm comparison for robustness

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your algorithm in `custom_retriever.py`
4. Test with `python src/main.py --algorithm your_algorithm --query_id 301`
5. Submit a pull request with performance results

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

## Contact

For questions about implementation or results, please open an issue in the repository.

---

**Last Updated**: January 2025  
**Current Status**: Active Development  
**Performance**: Baseline NDCG@10: 0.3505, Custom algorithms showing up to 25% variation
