# This document is to test the search functionality in the terminal by inputting a txt file with multiple queries.
import argparse
from main import search


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

for query in queries:
    print(f"Searching for query: {query}")
    results = search(query)
    for entry in results:
        entry['query'] = query
        entry['score'] = entry.get('score', 0)
    results.append({
        'query': query,
        'score': results[0]['score'] if results else 0,
        'results': [entry['url'] for entry in results]
    })

for entry in results:
    print("\n==============================")
    print(f"Query: {entry['query']}")
    print(f"Top Score: {entry['score']}")
    print("Results:")
    for i, url in enumerate(entry['results'], 1):
        print(f"  {i}. {url}")
print("\n==============================")