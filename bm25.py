import pickle


class BM25:

    def __init__(self):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.tf_data = self._load(self.path_to_TFs)
        self.idfs = self._load(self.path_to_IDFs)
    
    
    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print("[INFO] File not found.")
            return {}


    def _compute_doc_lengths(self, tf_data):
        return [sum(doc.values()) for doc in tf_data]


    def _bm25_score(self, query, doc_index, doc_lengths, avgdl, k1=1.5, b=0.75):
        score = 0.0
        doc_len = doc_lengths[doc_index]
        for term in query:
            tf = self.tf_data[doc_index].get(term.lower(), 0)
            idf = self.idfs.get(term, 0)    # TODO we could also add smoothing to deal with Out Of Vocabulary terms
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            if denom > 0:
                score += idf * ((tf * (k1 + 1)) / denom)
        return score


    def bm25_ranking(self, query):
        doc_lengths = self._compute_doc_lengths(self.tf_data)
        avgdl = sum(doc_lengths) / len(doc_lengths)

        scores = []
        for i in range(len(self.tf_data)):  # TODO check if it is right to use len(self.tf_data)
            score = self._bm25_score(query, i, doc_lengths, avgdl)
            scores.append((i, score))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)