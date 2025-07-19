import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from Utils.legal_crawling import OfflineCrawler
from Utils.indexer import Indexer
from Utils.bm25 import BM25
from Utils.hybrid_retrieval import HybridRetrieval

# Ensure stopwords are available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# TODO this function is taken mostly from crawling preprocessing function.
# i dont know if we should keep stopwords (think about negations in a query)
# or keep other information
def preprocess_query(text: str):
    # TODO how should we write tubingen? with ü or ue?
    # Add Tübingen in the query for related results
    text += " Tuebingen"

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters and normalize whitespace
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Filter stopwords and lemmatize words
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return filtered_tokens


initial_seeds = ["https://en.wikipedia.org/wiki/T%C3%BCbingen", 
                "https://uni-tuebingen.de/en/",
                "https://uni-tuebingen.de/en/research.html",
                "https://www.my-stuwe.de/en/",
                "https://www.germany.travel/en/cities-culture/tuebingen.html",
                "https://studieren.de/international-business-uni-tuebingen.studienprofil.t-0.a-68.c-110.html",
                "https://www.tuebingen.de/",
                "https://www.tuebingen.de/#"]

#     seeds = [
#         "https://visit-tubingen.co.uk/welcome-to-tubingen/",
#         "https://www.tuebingen.de/",
#         "https://uni-tuebingen.de/en/",
#         "https://en.wikipedia.org/wiki/T%C3%BCbingen",
#         "https://www.germany.travel/en/cities-culture/tuebingen.html",
#         "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
#         "https://www.europeanbestdestinations.com/destinations/tubingen/",
#         "https://www.tuebingen.de/en/",
#         "https://www.stadtmuseum-tuebingen.de/english/",
#         "https://www.tuebingen-info.de/",
#         "https://tuebingenresearchcampus.com/en/",
#         "https://www.welcome.uni-tuebingen.de/",
#         "https://integreat.app/tuebingen/en/news/tu-news",
#         "https://tunewsinternational.com/category/news-in-english/",
#         "https://historicgermany.travel/historic-germany/tubingen/",
#         "https://visit-tubingen.co.uk/",
#         "https://www.germansights.com/tubingen/",
#         "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
#         "https://www.tripadvisor.com/Restaurants-g198539-zfp58-Tubingen_Baden_Wurttemberg.html",
#         "https://www.tripadvisor.com/Attractions-g198539-Activities-c36-Tubingen_Baden_Wurttemberg.html",
#         "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany",
#         "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
#         "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
#         "https://simplifylivelove.com/tubingen-germany/",
#         "https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen",
#         "https://wanderlog.com/list/geoCategory/312176/best-spots-for-lunch-in-tubingen",
#         "https://guide.michelin.com/us/en/baden-wurttemberg/tbingen/restaurants",
#         "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
#         "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
#         "https://www.1821tuebingen.de/",
#         "https://www.historicgermany.travel/tuebingen/",
#         "https://www.expedia.com/Things-To-Do-In-Tubingen.d55289.Travel-Guide-Activities",
#         "https://www.lonelyplanet.com/germany/baden-wurttemberg/tubingen",
#         "https://www.tripadvisor.com/Attraction_Review-g198539-d14983273-Reviews-Tubingen_Weinwanderweg-Tubingen_Baden_Wurttemberg.html",
#         "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
#         "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
#         "https://wanderlog.com/geoInMonth/10053/7/tubingen-in-july",
#         "https://www.wanderlog.com/list/geoCategory/76026/best-restaurants-in-tubingen",
#         "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
#         "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
#         "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
#         "https://www.mygermanyvacation.com/things-to-do-in-tubingen/"
#     ]

""" 
crawler = OfflineCrawler(initial_seeds, max_depth=2)
crawler.run() """

crawler = OfflineCrawler(initial_seeds, max_depth=2)
crawler.run()

# ind = Indexer()
# ind.run()

# def search(query_text: str, use_hybrid_model=False):
#     query_tokens = preprocess_query(query_text)

#     candidates_ids = ind.get_union_candidates(query_tokens)

#     if not candidates_ids:
#             return []
     
#     if use_hybrid_model:
#         print("\nUsing hybrid model...\n")
#         model = HybridRetrieval()
#         return model.retrieve(query_tokens)
#     else:
#         print("\nUsing BM25 model...\n")
#         model = BM25()
#         return model.bm25_ranking(query_tokens, candidates_ids)


#for r in search('food', use_hybrid_model=True)[:10]:
#    print(r)

# TODO whats inside the pkl docs if i re run the crawler and indexer?
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer