from sentence_transformers import SentenceTransformer, util
import torch
from Utils.bm25 import BM25
from Utils.indexer import Indexer
import pandas as pd
import pickle


class HybridRetrieval:
    def __init__(self):
        self.bm25 = BM25()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate
        self.indexer = Indexer()  # For tokenizing queries etc.
        with open('data/sbert_doc_embeddings.pkl', 'rb') as f:
            self.doc_embeddings = pickle.load(f)  # {doc_id: tensor}


    def retrieve(self, query_tokens, candidate_doc_ids, top_k=50, lambda_bm25=0.5):
        """
        Perform two-stage retrieval: BM25 + SBERT re-ranking.
        """

        bm25_results = self.bm25.bm25_ranking(query_tokens, candidate_doc_ids)

        # Take top_k BM25 results
        top_k_urls = bm25_results[:top_k]

        # Prepare texts for semantic similarity
        doc_ids = []
        corpus_emb = []
        for doc_id in candidate_doc_ids:
            if doc_id in self.doc_embeddings:
                doc_ids.append(doc_id)
                corpus_emb.append(self.doc_embeddings[doc_id])

        # SBERT encoding
        query_text = " ".join(query_tokens)
        query_emb = self.model.encode(query_text, convert_to_tensor=True)
        
        device = query_emb.device
        corpus_emb = [torch.from_numpy(self.doc_embeddings[doc_id]).to(device) for doc_id in doc_ids]
        corpus_emb = torch.stack(corpus_emb)

        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_emb, corpus_emb)[0]
        

        # Combine BM25 and SBERT scores
        final_results = []
        for idx, doc_id in enumerate(doc_ids):
            url = self.bm25.crawled_data[doc_id]['url']
            bm25_score = top_k - idx  # Approximate rank-based BM25 score (normalize if needed)
            sbert_score = cosine_scores[idx].item()
            hybrid_score = lambda_bm25 * bm25_score + (1 - lambda_bm25) * sbert_score
            final_results.append((url, hybrid_score))

        # Sort by hybrid score
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)

        return [url for url, score in final_results]
