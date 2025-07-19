from Utils.legal_crawling import OfflineCrawler
from Utils.indexer import Indexer
from Utils.bm25 import BM25
from Utils.hybrid_retrieval import HybridRetrieval
from Utils.query_expander import QueryExpander
from Utils.text_preprocessor import preprocess_text


def search(query_text: str, use_hybrid_model=False, use_query_expansion=True):
    query_tokens = preprocess_text(query_text, isQuery=True)

    if use_query_expansion:
        print("\nExpanding query...\n")
        expander = QueryExpander(max_synonyms=2)
        expanded_tokens = expander.expand(query_tokens)
        print("Expanded query tokens:", expanded_tokens)
    else:
        expanded_tokens = query_tokens

    candidates_ids = ind.get_union_candidates(expanded_tokens)

    if not candidates_ids:
        return []

    if use_hybrid_model:
        print("\nUsing hybrid model...\n")
        model = HybridRetrieval()
        return model.retrieve(expanded_tokens)
    else:
        print("\nUsing BM25 model...\n")
        model = BM25()
        return model.bm25_ranking(expanded_tokens, candidates_ids)



initial_seeds = ["https://en.wikipedia.org/wiki/T%C3%BCbingen", 
                "https://uni-tuebingen.de/en/",
                "https://uni-tuebingen.de/en/research.html",
                "https://www.my-stuwe.de/en/",
                "https://www.germany.travel/en/cities-culture/tuebingen.html",
                "https://studieren.de/international-business-uni-tuebingen.studienprofil.t-0.a-68.c-110.html",
                "https://www.tuebingen.de/",
                "https://www.tuebingen.de/#"]

""" crawler = OfflineCrawler(initial_seeds, max_depth=2)
crawler.run() """

ind = Indexer()
#ind.run()


for r in search('chilling and called', use_hybrid_model=False, use_query_expansion=True)[:10]:
    print(r)

# TODO whats inside the pkl docs if i re run the crawler and indexer?
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer