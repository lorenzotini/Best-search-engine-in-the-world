import pickle

from Utils.indexer import Indexer


class BM25:

    def __init__(self):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.path_to_crawled_data = 'data/crawled_data.pkl'
        
        self.tf_data = self._load(self.path_to_TFs)
        self.idfs = self._load(self.path_to_IDFs)
        self.crawled_data = self._load(self.path_to_crawled_data)


    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print(f"[INFO] File {path} not found.")
            return {}


    def _compute_doc_lengths(self):
        return [sum(doc.values()) for doc in self.tf_data]


    def _bm25_score(self, weighted_query, doc_index, doc_lengths, avgdl, k1=1.5, b=0.75):
        score = 0.0
        doc_len = doc_lengths[doc_index]

        for term, weight in weighted_query:
            tf = self.tf_data[doc_index].get(term.lower(), 0)
            idf = self.idfs.get(term, 0)

            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            if denom > 0:
                score += weight * idf * ((tf * (k1 + 1)) / denom) 

        return score



    def bm25_ranking(self, weighted_query, candidate_doc_ids):
        doc_lengths = self._compute_doc_lengths()
        avgdl = sum(doc_lengths) / len(doc_lengths)

        scores = []
        ind = Indexer()

        for doc_id in candidate_doc_ids:
            score = self._bm25_score(weighted_query, doc_id, doc_lengths, avgdl)

            bonus = ind.proximity_bonus([term for term, _ in weighted_query], doc_id, window=3)
            score += bonus

            scores.append((doc_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        ranked_urls = []
        for doc_id, score in scores:
            url = self.crawled_data[doc_id]["url"]
            ranked_urls.append(url)

        return ranked_urls

