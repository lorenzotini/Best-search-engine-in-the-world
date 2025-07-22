from sentence_transformers import SentenceTransformer, util
import torch
from Utils.bm25 import BM25
from Utils.indexer import Indexer
import pickle


class HybridRetrieval:
    def __init__(self):
        self.bm25 = BM25()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate
        self.indexer = Indexer()
        with open('data/sbert_doc_embeddings.pkl', 'rb') as f:
            self.doc_embeddings = pickle.load(f)  # {doc_id: np.array}

    def retrieve(self, weighted_query, candidate_doc_ids, top_k=50, lambda_bm25=0.5):
        """
        Perform hybrid retrieval: BM25 retrieval + SBERT re-ranking.
        """

        # BM25 retrieval (returns list of tuples: (doc_id, bm25_score))
        bm25_results = self.bm25.bm25_ranking(weighted_query, candidate_doc_ids)

        # Keep only top_k BM25 results
        top_k_results = bm25_results[:top_k]

        # Prepare the query for SBERT (repeat terms based on their weights)
        query_text = " ".join([term for term, weight in weighted_query for _ in range(int(weight * 2))])
        query_emb = self.model.encode(query_text, convert_to_tensor=True)

        # Collect embeddings for top_k docs
        doc_ids = []
        corpus_emb = []
        for doc_id, bm25_score in top_k_results:
            if doc_id in self.doc_embeddings:
                doc_ids.append(doc_id)
                emb = torch.from_numpy(self.doc_embeddings[doc_id]).to(query_emb.device)
                corpus_emb.append(emb)

        if not corpus_emb:
            # Return BM25 results with original BM25 scores
            print("No embeddings found")
            return [(self.bm25.crawled_data[doc_id]['url'], bm25_score) for doc_id, bm25_score in top_k_results]

        corpus_emb = torch.stack(corpus_emb)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(query_emb, corpus_emb)[0]  # shape: (len(doc_ids),)

        # Hybrid scoring
        final_results = []
        bm25_score_dict = {doc_id: score for doc_id, score in top_k_results}

        for idx, doc_id in enumerate(doc_ids):
            url = self.bm25.crawled_data[doc_id]['url']
            bm25_score = bm25_score_dict.get(doc_id, 0)
            sbert_score = cosine_scores[idx].item()
            hybrid_score = lambda_bm25 * bm25_score + (1 - lambda_bm25) * sbert_score
            final_results.append((url, hybrid_score))

        # Sort by hybrid score descending
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results
