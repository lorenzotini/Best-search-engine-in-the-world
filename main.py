from Utils.legal_crawling import OfflineCrawler
from Utils.indexer import Indexer
from Utils.bm25 import BM25
from Utils.hybrid_retrieval import HybridRetrieval
from Utils.text_preprocessor import preprocess_text



def search(query_text: str, use_hybrid_model=False):

    query_tokens = preprocess_text(query_text, isQuery=True)

    candidates_ids = ind.get_union_candidates(query_tokens)

    if not candidates_ids:
            return []
     
    if use_hybrid_model:
        print("\nUsing hybrid model...\n")
        model = HybridRetrieval()
        return model.retrieve(query_tokens)
    else:
        print("\nUsing BM25 model...\n")
        model = BM25()
        return model.bm25_ranking(query_tokens, candidates_ids)




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
ind.run()


for r in search('improving changed', use_hybrid_model=False)[:10]:
    print(r)

# TODO whats inside the pkl docs if i re run the crawler and indexer?
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer