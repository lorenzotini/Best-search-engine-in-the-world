# This document is to test the search functionality in the terminal by inputting a txt file with multiple queries.
import argparse
from main import search, init_search
import time

indexer, bm25_model, hybrid_model, sentiment_pipeline = init_search()

start = time.time()

parser = argparse.ArgumentParser(
    prog="search.py",
    description="Test the search functionality with multiple queries from a text file.")

parser.add_argument("--filename", type=str, required=True, help="Path to the text file containing queries.")

args = parser.parse_args()

queries = []
with open(args.filename, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            queries.append(parts[1])

print("Loaded queries:", queries)

results = []

end = time.time()

print(f"Time to load queries: {end - start:.2f} seconds")
for query in queries:
    print(f"Searching for query: {query}")
    results_query = search(query, indexer, bm25_model, hybrid_model, use_hybrid_model=True, use_query_expansion=True)
    results.append({
        'query': query,
        'results': [{'url': entry[0], 'score': entry[1]} for entry in results_query[:10]]})
    
for entry in results:
    print("\n============================================================")
    print(f"Query: {entry['query']}")
    print("Results:")
    print(f"{'URL':<100} {'SCORE':>10}")
    print("-" * 72)
    for result in entry['results']:
        print(f"{result['url']:<100} {result['score']:>10}")
print("\n============================================================")