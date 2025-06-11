#!/bin/bash
poetry env activate

echo "--------------------------------"
echo " Running Baseline Algorithms"
echo "--------------------------------"
echo "BASELINE: Running pyterrier bm25 with no stopwords"
poetry run python3 src/main.py --retriever pyterrier --algorithm bm25 --k 20 --no-stopwords --rebuild-index --save-results

echo "BASELINE: Running pyterrier bm25 with stopwords"
poetry run python3 src/main.py --retriever pyterrier --algorithm bm25 --k 20 --use-stopwords --rebuild-index --save-results

echo "--------------------------------"
echo " Running Custom Algorithms"
echo "--------------------------------"
echo "CUSTOM: Running custom bm25 with no stopwords"
poetry run python3 src/main.py --retriever custom --algorithm bm25 --k 20 --save-results
poetry run python3 src/main.py --retriever custom --algorithm bm25_plus --k 20 --save-results
poetry run python3 src/main.py --retriever custom --algorithm tf_idf --k 20 --save-results
poetry run python3 src/main.py --retriever custom --algorithm cosine --k 20 --save-results
poetry run python3 src/main.py --retriever custom --algorithm language_model --k 20 --save-results
poetry run python3 src/main.py --retriever custom --algorithm financial_boost --k 20 --save-results

# Read the results/ directory and print the results in a table format
echo "--------------------------------"
echo " Results"
echo "--------------------------------" | tee results.txt
for file in results/*.json; do
    echo "--------------------------------" | tee -a results.txt
    echo " $file" | tee -a results.txt
    echo "--------------------------------" | tee -a results.txt
    cat $file | jq -r 'to_entries[] | "\(.key): \(.value)"'
done | tee -a results.txt