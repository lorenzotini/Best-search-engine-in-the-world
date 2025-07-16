import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from crawling import OfflineCrawler
from indexer import Indexer
from bm25 import BM25


# Ensure stopwords are available
import nltk
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

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



seeds = ["https://en.wikipedia.org/wiki/T%C3%BCbingen"]

crawler = OfflineCrawler(seeds, max_depth=2)
crawler.run()

""" indexer = Indexer()
indexer.run()
 """

""" query = 'home'
query = preprocess_query(query)

model = BM25()
ranking = model.bm25_ranking(query)

for doc_idx, score in ranking:
    print(f"Document {doc_idx}: BM25 score = {score:.4f}") """


# TODO whats inside the pkl docs if i re run the crawler and indexer?

# TODO sembra che bm25 non funzioni: alla query Traffic il primo risultato manco contiene il termine
# TODO salvare il log del crawling
# TODO stampare solo le prime 10 results del ranking
# TODO controllare che non sparisca il log file: ora scompare se runno crawl, interrupt, run indexer