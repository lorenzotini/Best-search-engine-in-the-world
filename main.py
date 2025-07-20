from Utils.legal_crawling import OfflineCrawler
from Utils.indexer import Indexer
from Utils.bm25 import BM25
from Utils.hybrid_retrieval import HybridRetrieval
from Utils.query_expander import QueryExpander
from Utils.text_preprocessor import preprocess_text
import time


def search(query_text: str, use_hybrid_model=False, use_query_expansion=True):
    start = time.time()
    query_tokens = preprocess_text(query_text, isQuery=True)
    end = time.time()
    print("Time to preprocess query: ", end - start)
    
    start = time.time()
    if use_query_expansion:
        print("\nExpanding query...\n")
        expander = QueryExpander(max_synonyms=2)
        expanded_tokens = expander.expand(query_tokens)
        print("Expanded query tokens:", expanded_tokens)
    else:
        expanded_tokens = query_tokens
    end = time.time()
    print("Time to expand query: ", end - start)

    start = time.time()
    ind = Indexer(silent=True)
    candidates_ids = ind.get_union_candidates(expanded_tokens)
    end = time.time()
    print("Time to get candidates: ", end - start)

    if not candidates_ids:
        return []
    
    start = time.time()
    if use_hybrid_model:
        print("\nUsing hybrid model...\n")
        model = HybridRetrieval()
        results = model.retrieve(expanded_tokens)
    else:
        print("\nUsing BM25 model...\n")
        model = BM25()
        results = model.bm25_ranking(expanded_tokens, candidates_ids)
    end = time.time()
    print("Time to rank: ", end - start)

    return results

seeds = [
    "https://visit-tubingen.co.uk/welcome-to-tubingen/",
    "https://www.tuebingen.de/",
    "https://uni-tuebingen.de/en/",
    "https://en.wikipedia.org/wiki/T%C3%BCbingen",
    "https://www.germany.travel/en/cities-culture/tuebingen.html",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
    "https://www.europeanbestdestinations.com/destinations/tubingen/",
    "https://www.tuebingen.de/en/",
    "https://www.stadtmuseum-tuebingen.de/english/",
    "https://www.tuebingen-info.de/",
    "https://tuebingenresearchcampus.com/en/",
    "https://www.welcome.uni-tuebingen.de/",
    "https://integreat.app/tuebingen/en/news/tu-news",
    "https://tunewsinternational.com/category/news-in-english/",
    "https://historicgermany.travel/historic-germany/tubingen/",
    "https://visit-tubingen.co.uk/",
    "https://www.germansights.com/tubingen/",
    "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
    "https://www.tripadvisor.com/Restaurants-g198539-zfp58-Tubingen_Baden_Wurttemberg.html",
    "https://www.tripadvisor.com/Attractions-g198539-Activities-c36-Tubingen_Baden_Wurttemberg.html",
    "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany",
    "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
    "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
    "https://simplifylivelove.com/tubingen-germany/",
    "https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen",
    "https://wanderlog.com/list/geoCategory/312176/best-spots-for-lunch-in-tubingen",
    "https://guide.michelin.com/us/en/baden-wurttemberg/tbingen/restaurants",
    "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
    "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
    "https://www.1821tuebingen.de/",
    "https://www.historicgermany.travel/tuebingen/",
    "https://www.expedia.com/Things-To-Do-In-Tubingen.d55289.Travel-Guide-Activities",
    "https://www.lonelyplanet.com/germany/baden-wurttemberg/tubingen",
    "https://www.tripadvisor.com/Attraction_Review-g198539-d14983273-Reviews-Tubingen_Weinwanderweg-Tubingen_Baden_Wurttemberg.html",
    "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
    "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
    "https://wanderlog.com/geoInMonth/10053/7/tubingen-in-july",
    "https://www.wanderlog.com/list/geoCategory/76026/best-restaurants-in-tubingen",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
    "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
    "https://www.mygermanyvacation.com/things-to-do-in-tubingen/"
]


""" crawler = OfflineCrawler(seeds, max_depth=2)
crawler.run() """

start = time.time()
#ind = Indexer()
#ind.run()
end = time.time()
print("Time to index: ", end - start)

for r in search('food', use_hybrid_model=False, use_query_expansion=True)[:10]:
    print(r)



# TODO implement query term relevance: query terms from the user must be 
# prioritiezed over expanded query terms