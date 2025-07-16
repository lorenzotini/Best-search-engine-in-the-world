# indexer.py

import os
import pickle
import math
from collections import defaultdict
from index import can_skip, intersect_skip, in_range, intersect_range

class Indexer:
    def __init__(self):
        # File paths in current directory
        self.path_to_crawled_data  = 'crawled_data.pkl'
        self.path_to_posting_lists = 'posting_list.pkl'
        self.path_to_TFs           = 'tfs.pkl'
        self.path_to_IDFs          = 'idfs.pkl'

        # Load crawled data
        if not os.path.exists(self.path_to_crawled_data):
            print(f"[ERROR] No crawl data at {self.path_to_crawled_data}. Aborting indexing.")
            self.crawled_data = {}
        else:
            with open(self.path_to_crawled_data, 'rb') as f:
                self.crawled_data = pickle.load(f)

        # Load or initialize posting lists
        if os.path.exists(self.path_to_posting_lists):
            with open(self.path_to_posting_lists, 'rb') as f:
                self.skip_dict, self.pos_index_dict = pickle.load(f)
        else:
            self.skip_dict, self.pos_index_dict = {}, {}

    def run(self):
        self._build_skip_pointers()
        self._build_positional_index()
        self._save_posting_lists()
        self._build_TF()
        self._build_IDF()

    def _build_skip_pointers(self):
        token_to_ids = {}
        # Collect document lists per token
        for doc_id, doc in self.crawled_data.items():
            toks = doc.get('tokens') or []
            for t in toks:
                token_to_ids.setdefault(t, []).append(doc_id)
        # Build skip lists
        skip = {}
        for token, ids in token_to_ids.items():
            freq = math.ceil(math.sqrt(len(ids)))
            pointers = []
            for i, d in enumerate(ids):
                skip_to = None
                skip_doc = None
                if i % freq == 0:
                    idx = min(i + freq, len(ids)-1)
                    skip_to = idx
                    skip_doc = ids[idx]
                pointers.append([d, skip_to, skip_doc])
            skip[token] = pointers
        self.skip_dict = skip

    def _build_positional_index(self):
        idx = defaultdict(lambda: defaultdict(list))
        for doc_id, doc in self.crawled_data.items():
            toks = doc.get('tokens') or []
            for pos, t in enumerate(toks):
                idx[t][doc_id].append(pos)
        # convert to lists
        final = {t: [[d, ps] for d, ps in docs.items()] for t, docs in idx.items()}
        self.pos_index_dict = final

    def _save_posting_lists(self):
        with open(self.path_to_posting_lists, 'wb') as f:
            pickle.dump((self.skip_dict, self.pos_index_dict), f)
        print("[INFO] Saved posting_list.pkl")

    def _build_TF(self):
        tf_list = []
        for doc in self.crawled_data.values():
            bow = {}
            for w in doc.get('tokens') or []:
                bow[w] = bow.get(w, 0) + 1
            tf_list.append(bow)
        with open(self.path_to_TFs, 'wb') as f:
            pickle.dump(tf_list, f)
        print("[INFO] Saved tfs.pkl")

    def _build_IDF(self):
        N = len(self.crawled_data)
        df = {}
        # document frequency
        for doc in self.crawled_data.values():
            seen = set(doc.get('tokens') or [])
            for w in seen:
                df[w] = df.get(w, 0) + 1
        idfs = {w: math.log(N/df[w], 10) for w in df}
        with open(self.path_to_IDFs, 'wb') as f:
            pickle.dump(idfs, f)
        print("[INFO] Saved idfs.pkl")
