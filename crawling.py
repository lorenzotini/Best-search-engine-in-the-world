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
import math
from collections import defaultdict
from langdetect import detect_langs


# Ensure stopwords are available
import nltk
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")



stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class OfflineCrawler:
    """Manually triggered crawler: from seed URLs, collect page texts."""
    def __init__(self, seeds, max_depth=2, delay=0.5):
        self.seeds = seeds
        self.max_depth = max_depth
        self.delay = delay

        self.path_to_crawled_data = 'crawled_data.pkl'
        self.crawled_data = self._load(self.path_to_crawled_data)
        
        self.path_to_posting_lists = 'posting_list.pkl'
        self.skip_dict, self.pos_index_dict = self._load(self.path_to_posting_lists)
        
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
        
        print('Indexing...')
        self.crawled_data = self._load(self.path_to_crawled_data)
        self._build_skip_pointers(self.crawled_data)
        self._build_positional_index(self.crawled_data)
        
        print("Saving posting lists...")
        self._save_posting_lists()
        print("Done.")


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
            # TODO returnare qualcosa
            return
        
        # Lowercase
        text = text.lower()

        # Remove non-alphabetic characters and normalize whitespace
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Filter stopwords and stem words
        filtered_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in stop_words and len(token) > 2
        ]

        return filtered_tokens


    def _build_skip_pointers(self, crawled_data):
        token_to_ids = {}
        skip_dict = {}

        # Process every document
        for doc_id, tokens in crawled_data.items():

            # For every token, create a list containing the IDs of the documents in which the token is present
            for token in tokens:
                if token_to_ids.get(token) is None:    # First time we see this term -> initialize it
                    token_to_ids[token] = [doc_id]
                elif doc_id in token_to_ids[token]:
                    continue
                else:
                    token_to_ids[token].append(doc_id)
        
        # Add skip pointers
        for token in token_to_ids:
            current_ids = token_to_ids[token]
            
            # Insert a pointer every pointer_freq entry
            pointer_freq = math.ceil(math.sqrt(len(token_to_ids[token])))

            skip_pointers = []
            for i, id in enumerate(current_ids):
                index_to_skip = None
                doc_at_that_index = None
                if i % pointer_freq == 0:
                    index_to_skip = i + pointer_freq
                    if index_to_skip >= len(current_ids):
                        index_to_skip = len(current_ids) - 1
                    doc_at_that_index = current_ids[index_to_skip]
                skip_pointers.append([id, index_to_skip, doc_at_that_index])
            
            skip_dict[token] = skip_pointers
        
        self.skip_dict = skip_dict

    
    def _build_positional_index(self, crawled_data):

        index = defaultdict(lambda: defaultdict(list))  # token -> {doc_id: [positions]}

        for doc_id, tokens in crawled_data.items():
            for position, token in enumerate(tokens):
                index[token][doc_id].append(position)

        # Convert inner dicts to lists for final output format
        final_index = {}
        for token, doc_dict in index.items():
            final_index[token] = [[doc_id, positions] for doc_id, positions in doc_dict.items()]

        self.pos_index_dict = final_index

    def _save_crawled(self, doc_id, tokens):
        self.crawled_data.update({doc_id : tokens})

        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)

    def _save_posting_lists(self):
        with open(self.path_to_posting_lists, "wb") as f:
            pickle.dump((self.skip_dict, self.pos_index_dict), f)

    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            if path == self.path_to_crawled_data:
                print("[INFO] No previous crawl data found, starting fresh.")
                return {}
            elif path == self.path_to_posting_lists:
                print("[INFO] No existing posting lists found, will build from scratch.")
                return {}, {}
            else:
                raise  # re-raise for unknown files

    

seeds = ["https://example.com"]
crawler = OfflineCrawler(seeds, max_depth=1)
pages = crawler.run()

post = crawler._load('posting_list.pkl')
a = 0