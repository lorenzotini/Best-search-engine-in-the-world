from sentence_transformers import SentenceTransformer, util
from Utils.bm25 import BM25
from Utils.indexer import Indexer
import pandas as pd
from IPython.display import display


class HybridRetrieval:
    def __init__(self):
        self.bm25 = BM25()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate
        self.indexer = Indexer()  # For tokenizing queries etc.

    def retrieve(self, query_tokens, top_k=50, lambda_bm25=0.5):
        """
        Perform two-stage retrieval: BM25 + SBERT re-ranking.
        """

        # Get candidate doc IDs from BM25
        candidate_doc_ids = list(self.bm25.crawled_data.keys())
        bm25_results = self.bm25.bm25_ranking(query_tokens, candidate_doc_ids)

        # Take top_k BM25 results
        top_k_urls = bm25_results[:top_k]

        # Prepare texts for semantic similarity
        corpus_texts = []
        doc_ids = []
        for doc_id in candidate_doc_ids:
            url = self.bm25.crawled_data[doc_id]['url']
            if url in top_k_urls:
                doc_ids.append(doc_id)
                # Reconstruct document text for SBERT (join tokens)
                tokens = self.bm25.crawled_data[doc_id]['tokens']
                corpus_texts.append(" ".join(tokens))

        # SBERT encoding
        query_text = " ".join(query_tokens)
        query_emb = self.model.encode(query_text, convert_to_tensor=True)
        corpus_emb = self.model.encode(corpus_texts, convert_to_tensor=True)

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


    def visualize_results(self, query_text, top_k=10, lambda_bm25=0.5):
        from main import preprocess_query
        query_tokens = preprocess_query(query_text)

        candidate_doc_ids = list(self.bm25.crawled_data.keys())
        bm25_results = self.bm25.bm25_ranking(query_tokens, candidate_doc_ids)
        top_k_urls = bm25_results[:top_k]

        corpus_texts = []
        doc_ids = []
        for doc_id in candidate_doc_ids:
            url = self.bm25.crawled_data[doc_id]['url']
            if url in top_k_urls:
                doc_ids.append(doc_id)
                tokens = self.bm25.crawled_data[doc_id]['tokens']
                corpus_texts.append(" ".join(tokens))

        query_emb = self.model.encode(query_text, convert_to_tensor=True)
        corpus_emb = self.model.encode(corpus_texts, convert_to_tensor=True)

        cosine_scores = util.cos_sim(query_emb, corpus_emb)[0]

        # Build DataFrame for visualization
        data = []
        for idx, doc_id in enumerate(doc_ids):
            url = self.bm25.crawled_data[doc_id]['url']
            bm25_rank = top_k_urls.index(url) + 1
            bm25_score = top_k - bm25_rank + 1  # Simple rank-based proxy
            sbert_score = cosine_scores[idx].item()
            hybrid_score = lambda_bm25 * bm25_score + (1 - lambda_bm25) * sbert_score

            snippet = " ".join(self.bm25.crawled_data[doc_id]['tokens'][:20]) + "..."

            data.append({
                "Rank": idx + 1,
                "URL": url,
                "BM25_Score": bm25_score,
                "SBERT_Cosine": round(sbert_score, 3),
                "Hybrid_Score": round(hybrid_score, 3),
                "Snippet": snippet
            })

        df = pd.DataFrame(data)
        display(df)
        return df
