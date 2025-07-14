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


from main import preprocess_query

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

    
class Indexer:
    def __init__(self):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_posting_lists = 'data/posting_list.pkl'
        
        self.crawled_data = self._load(self.path_to_crawled_data)
        self.skip_dict, self.pos_index_dict = self._load(self.path_to_posting_lists)


    def index_documents(self):
        print('Indexing...')
        self._build_skip_pointers(self.crawled_data)
        self._build_positional_index(self.crawled_data)
        
        print("Saving posting lists...")
        with open(self.path_to_posting_lists, "wb") as f:
            pickle.dump((self.skip_dict, self.pos_index_dict), f)
        print("Done.")


    def _build_skip_pointers(self, crawled_data):
        token_to_ids = {}
        skip_dict = {}

        # Process every document
        for doc_id, tokens in crawled_data.items():

            # For every token, create a list containing the IDs of the documents in which the token is present
            for token in tokens:
                if token not in token_to_ids:    # First time we see this term -> initialize it
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


    def build_TF(self):
        tf_list = []
        for id, tokens in self.crawled_data.items():            
            # build term-frequency dictionary
            bow = {}
            for word in tokens:
                bow[word] = bow.get(word, 0) + 1
            
            tf_list.append(bow)
        # Save data in file
        with open(self.path_to_TFs, "wb") as f:
            pickle.dump(tf_list, f)


    #Estimate inverse document frequencies based on a corpus of documents.
    def build_IDF(self):
        idfs = {}
        D = len(self.crawled_data)
        # TODO remove this printf after verified that it is correct
        print("\nNumber of document in the corpus: ", D, "\ndelete this print pls\n")
        # Dts = {key: str, value: (Dt_value: int, seen_in_this_doc: bool)}
        Dts = {}
        for doc_id, tokens in self.crawled_data.items(): # TODO .items()?

            # Reset visited values, since we are visiting a new document
            for key, (num, _) in Dts.items():
                Dts[key] = (num, False)

            for word in tokens:
                if Dts.get(word) is None:    # First time we see this term -> initialize it
                    Dts[word] = (1, True)
                elif Dts[word][1] is False:  # Dt already contains an entry for this term but is the first time we see it in this doc
                    Dts[word] = (Dts[word][0] + 1, True)   # Increment its frequency and toggle bool
                elif Dts[word][1] is True:
                    continue                # We've already seen this term in this doc
                else:
                    raise Exception('Something went wrong during building idfs.')
        
        for k, v in Dts.items():
            Dt = v[0]
            idfs[k] = math.log(D/Dt, 10)
            
        # Save data in file
        with open(self.path_to_IDFs, "wb") as f:
            pickle.dump(idfs, f)


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


class BM25:

    def __init__(self):
        self.path_to_TFs = 'data/tfs.pkl'
        self.path_to_IDFs = 'data/idfs.pkl'
        self.tf_data = self._load(self.path_to_TFs)
        self.idfs = self._load(self.path_to_IDFs)
    
    
    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print("[INFO] File not found.")
            return {}


    def compute_doc_lengths(self, tf_data):
        return [sum(doc.values()) for doc in tf_data]


    def bm25_score(self, query, doc_index, doc_lengths, avgdl, k1=1.5, b=0.75):
        score = 0.0
        doc_len = doc_lengths[doc_index]
        for term in query:
            tf = self.tf_data[doc_index].get(term.lower(), 0)
            idf = self.idfs.get(term, 0)    # TODO we could also add smoothing to deal with Out Of Vocabulary terms
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            if denom > 0:
                score += idf * ((tf * (k1 + 1)) / denom)
        return score


    def bm25_ranking(self, query):
        doc_lengths = self.compute_doc_lengths(self.tf_data)
        avgdl = sum(doc_lengths) / len(doc_lengths)

        scores = []
        for i in range(len(self.tf_data)):  # TODO check if it is right to use len(self.tf_data)
            score = self.bm25_score(query, i, doc_lengths, avgdl)
            scores.append((i, score))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)


########################################################################################

seeds = ["https://example.com"]
crawler = OfflineCrawler(seeds, max_depth=1)
crawler.run()
indexer = Indexer()
indexer.index_documents()
indexer.build_TF()
indexer.build_IDF()

query = 'traffic'
query = preprocess_query(query)

model = BM25()

ranking = model.bm25_ranking(query)

for doc_idx, score in ranking:
    print(f"Document {doc_idx}: BM25 score = {score:.4f}")