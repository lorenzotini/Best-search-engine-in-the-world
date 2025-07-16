#!/usr/bin/env python3
# main.py

import os
import pickle
from dynamic_domain_crawler import DynamicDomainCrawler
from indexer import Indexer
from index import intersect_skip, intersect_range

def inspect_pickles():
    """Load and print basic info about all generated pickle files."""
    def info(path):
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            return
        obj = pickle.load(open(path, 'rb'))
        size = len(obj) if hasattr(obj, '__len__') else 'unknown'
        print(f"[OK] {path}: {size}")
    print("\n=== Inspecting Pickle Outputs ===")
    info('crawled_data.pkl')
    info('simhashes.pkl')
    info('frontier.pkl')
    info('posting_list.pkl')
    info('tfs.pkl')
    info('idfs.pkl')
    print("=================================\n")

def test_posting_list():
    """Perform a quick posting-list sanity check."""
    print("=== Testing posting-list functions ===")
    skip_dict, pos_index = pickle.load(open('posting_list.pkl','rb'))
    term = next(iter(pos_index.keys()), None)
    if term:
        docs = [entry[0] for entry in pos_index[term]]
        print(f"Term '{term}' appears in docs: {docs[:5]}{'...' if len(docs)>5 else ''}")
        # test intersect_range on first two docs if available
        if len(docs) >= 2:
            positions1 = pos_index[term][0][1]
            positions2 = pos_index[term][1][1]
            rng = 2
            intr = intersect_range(positions1, positions2, rng)
            print(f"Positions in doc {pos_index[term][0][0]} within ±{rng} of doc {pos_index[term][1][0]}: {intr}")
    else:
        print("No terms in positional index to test.")
    print("======================================\n")

def main():
    # 1) Crawl
    seeds = [
        'https://www.germany.travel/en/cities-culture/tuebingen.html',
        'https://uni-tuebingen.de',
        'https://de.wikipedia.org/wiki/Tübingen'
    ]
    crawler = DynamicDomainCrawler(
        seeds,
        max_depth=3,
        simhash_threshold=3,
        max_threads=10,
        time_limit=2,
        max_new_pages=5
    )
    crawler.run()

    # 2) Index
    indexer = Indexer()
    indexer.run()

    # 3) Inspect outputs
    inspect_pickles()

    # 4) Test posting-list logic
    test_posting_list()

if __name__ == '__main__':
    main()
