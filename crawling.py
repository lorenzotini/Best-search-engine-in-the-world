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
import logging
import sys
import signal
import os


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


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("log/crawler.log", mode='w', encoding='utf-8'),  # Save logs to file (overwrite)
        logging.StreamHandler()  # Print logs to console
    ]
)


class OfflineCrawler:
    """Manually triggered crawler: from seed URLs, collect page texts."""
    def __init__(self, seeds, max_depth=2, delay=0.5):
        self.seeds = seeds
        self.max_depth = max_depth
        self.delay = delay

        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.crawled_data = self._load(self.path_to_crawled_data)
        signal.signal(signal.SIGINT, self._handle_interrupt)        # USE CTR+C TO INTERRUPT THE CRAWLING, NOT CTR+Z THAT MIGHT CORRUPT THE FILE

        

    def run(self):
        frontier = deque([(url, 0) for url in self.seeds])
        visited = set()
        non_english_domains = (".de", ".fr", ".es", ".ru", ".cn", ".it", ".pl", ".jp", ".br")


        while frontier:
            url, depth = frontier.popleft()
            if url in visited or depth > self.max_depth:
                continue
            visited.add(url)

            try:
                #resp = requests.get(url, timeout=5)
                resp = requests.get(url, headers={"Accept-Language": "en-US,en;q=0.9"}, timeout=5)
                resp.raise_for_status()
                html = resp.text
            except Exception as e:
                logging.error(f"fetching {url}: {e}")
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            
            tokens = self._preprocess_text(soup)
            if tokens is None:
                continue  # Skip saving and linking for non-English pages
            
            doc_id = self._get_id(url)
            logging.info(f"depth: {depth} fetching id: {doc_id}, {url}")
            # Just save the data and not compute the posting list, so it can be done at the end of the crawling
            # process only one time (useful for skip pointers). Also, disconnection resistant
            self._save_crawled(doc_id, tokens)


            # enqueue links if depth allows
            if depth < self.max_depth:
                
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    href, _ = urldefrag(href)
                    p = urlparse(href)

                    # Skip non-HTTP links
                    if p.scheme not in ("http", "https"):
                        continue

                    # Heuristic 1: Skip known non-English domains
                    if any(href.endswith(domain) for domain in non_english_domains):
                        continue

                    # Heuristic 2: Skip URLs with language codes in the path
                    if re.search(r"/(de|fr|es|ru|cn|it|pl|jp|br)(/|$)", href):
                        continue

                    frontier.append((href, depth + 1))

            time.sleep(self.delay)


    # TODO this is randomly done, make something useful
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
            logging.warning("non-English page, skipping.")
            # TODO return something
            return None
        
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
        self.crawled_data[doc_id] = tokens

        tmp_path = self.path_to_crawled_data + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.crawled_data, f)

        os.replace(tmp_path, self.path_to_crawled_data)  # atomic rename


    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            logging.info("No previous crawl data found, starting fresh.")
            return {}


    def _handle_interrupt(self, signum, frame):
        logging.info("Interrupted! Saving current progress...")
        self._save_all()
        sys.exit(0)


    def _save_all(self):
        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)