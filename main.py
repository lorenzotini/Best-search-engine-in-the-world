import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from legal_crawling import OfflineCrawler
from indexer import Indexer
from bm25 import BM25

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
def preprocess_query(text):
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


    
initial_seeds = ["https://en.wikipedia.org/wiki/T%C3%BCbingen"]
seeds = ["https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html"] # Replace with actual seeds

""" crawler = OfflineCrawler(initial_seeds, max_depth=2)
crawler.run()  """

ind = Indexer()
ind.run()

query = 'building agriculture'
query = preprocess_query(query)

candidates_ids = ind.get_candidates(query, use_proximity=False)

model = BM25()
ranking = model.bm25_ranking(query, candidates_ids)

for url in ranking:
    print(url)

# TODO whats inside the pkl docs if i re run the crawler and indexer?
# TODO sembra che bm25 non funzioni: alla query Traffic il primo risultato manco contiene il termine
# TODO stampare solo le prime 10 results del ranking
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer