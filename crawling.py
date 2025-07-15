import time
import requests
from urllib.parse import urljoin, urldefrag, urlparse
from nltk.corpus import stopwords
from collections import deque
import pickle
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs


# Ensure stopwords are available
import nltk
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")




stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class OfflineCrawler:
    """Manually triggered crawler: from seed URLs, collect page texts."""
    def __init__(self, seeds, max_depth=2, delay=0.5):
        self.seeds = seeds
        self.max_depth = max_depth
        self.delay = delay

        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.crawled_data = self._load(self.path_to_crawled_data)
        

    def run(self):
        frontier = deque([(url, 0) for url in self.seeds])
        visited = set()

        while frontier:
            url, depth = frontier.popleft()
            if url in visited or depth > self.max_depth:
                continue
            visited.add(url)

            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                html = resp.text
            except Exception as e:
                print(f"[ERROR] fetching {url}: {e}")
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            tokens = self._preprocess_text(soup)
            doc_id = self._get_id(url)
            print(f"[INFO] fetching id: {doc_id}, {url}")
            # Just save the data and not compute the posting list, so it can be done at the end of the crawling
            # process only one time (useful for skip pointers). Also, disconnection resistant
            self._save_crawled(doc_id, tokens)


            # enqueue links if depth allows
            if depth < self.max_depth:
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    href, _ = urldefrag(href)
                    p = urlparse(href)
                    if p.scheme in ("http", "https"):
                        frontier.append((href, depth + 1))

            time.sleep(self.delay)


    def _get_id(self, url):
        if self.crawled_data:
            return max(self.crawled_data.keys()) + 1
        return 0


    def _preprocess_text(self, soup):
        # Extract visible text
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")

        # Language filter
        langs = []
        try:
            langs = detect_langs(text)
        except:
            pass
        if not any(l.lang == "en" and l.prob >= 0.9 for l in langs):
            print(f"[SKIP] non-English")
            # TODO return something
            return
        
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


    def _save_crawled(self, doc_id, tokens):
        self.crawled_data.update({doc_id : tokens})

        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)


    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print("[INFO] No previous crawl data found, starting fresh.")
            return {}
