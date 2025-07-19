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


""" crawler = OfflineCrawler(initial_seeds, max_depth=2)
crawler.run() """

ind = Indexer()
ind.run()

def search(query_text: str, use_hybrid_model=False):
    query_tokens = preprocess_query(query_text)

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


for r in search('food', use_hybrid_model=True)[:10]:
    print(r)

# TODO whats inside the pkl docs if i re run the crawler and indexer?
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer