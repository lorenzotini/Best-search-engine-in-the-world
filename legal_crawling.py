import sys
import time
import requests
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse 
import nltk

from bm25 import BM25
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
import pickle
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs
import heapq 
import urllib.robotparser 
import hashlib
import logging
import signal
from indexer import Indexer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("log/crawler.log", mode='a', encoding='utf-8'),  # Save logs to file (overwrite)
        logging.StreamHandler()  # Print logs to console
    ]
)

class OfflineCrawler:
    def __init__(self, seeds, max_depth=2, delay=0.5, simhash_threshold=3):
        self.seeds = seeds
        self.max_depth = max_depth
        self.default_delay = delay # Store the default delay separately
        self.simhash_threshold = simhash_threshold

        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_simhashes = 'data/simhashes.pkl'
        
        self.crawled_data = self._load(self.path_to_crawled_data)
        self.seen_simhashes = self._load(self.path_to_simhashes)
        
        signal.signal(signal.SIGINT, self._handle_interrupt) 
        
        # Priority queue for frontier: (priority, url, depth)
        self.frontier = []
        # Keep track of URLs that have been added to the frontier to avoid duplicates
        self.visited_urls_in_queue = set()

        # For mapping URL to doc_id (helpful for debugging and future relevance)
        self.url_to_doc_id = {url_info['url']: doc_id for doc_id, url_info in self.crawled_data.items() if 'url' in url_info}

        # NEW: Robots.txt Parsers Cache and User-Agent
        # Stores RobotFileParser objects, keyed by domain (netloc)
        self.robot_parsers = {}
        # Stores the effective crawl delay for each domain
        self.domain_delays = {}
        # User-Agent string for your crawler. Be descriptive and include a contact URL.
        # This is what websites will see in their logs and what robots.txt rules apply to.
        # IMPORTANT: REPLACE with your actual contact URL!
        self.user_agent = "TübingenSearchBot_UniProject/4.20 (https://alma.uni-tuebingen.de/alma/pages/startFlow.xhtml?_flowId=detailView-flow&unitId=78284&periodId=228&navigationPosition=studiesOffered,searchCourses)" 
            
            # NEW: Tiered keywords for weighted priority
        self.very_relevant_keywords = {
            "tuebingen", "tubingen", "eberhard karls universität", "hohentuebingen", "hohentübingen",
            "stocherkahnrennen", "chocolart", "university", "universität", "neckar", "altstadt", "baden-wuerttemberg", "baden-württemberg"
        }
        self.relevant_keywords = {
            "city", "stadtfest", "cyber valley", "uniklinik", "university hospital", "tourism", "visitors",
            "restaurants", "cafes", "gastronomy", "food", "drinks", "bars", "bakery",
            "butcher", "market", "wochenmarkt","schwaebische alb", "schwübische alb", "schoenbuch", "schönbuch",
            "umbrisch-provenzalischer markt","jazz & klassik tage", "sommernachtskino"
        }
        self.moderately_relevant_keywords = {
            "community", "information", "news", "events", "travel", "sights", "attractions", "living",
            "citizens", "local", "region", "research", "science", "study", "students",
            "reutlingen", "rottenburg am neckar", "moessingen", "mössingen",
            "kirchentellinsfurt", "ammerbuch", "dusslingen", "dußlingen", "gomaringen",
            "kusterdingen", "ofterdingen", "nehren", "dettenhausen",
            "housing", "living", "real estate", "traffic", "public transport", "parking",
            "health", "doctors", "pharmacies", "schools", "kindergartens", "sport",
            "clubs", "associations", "town hall", "administration", "waste", "building",
            "culture", "art", "theater", "theatre", "museums", "exhibitions", "music",
            "cinema", "books", "literature", "galleries"
        }


    def run(self):
        self.last_fetch_time = {} 

        for url in self.seeds:
            if url not in self.visited_urls_in_queue:
                rp = self._get_robot_parser(url)
                if rp.can_fetch(self.user_agent, url):
                    priority = self._calculate_priority(url, "", [], 0)
                    heapq.heappush(self.frontier, (priority, url, 0))
                    self.visited_urls_in_queue.add(url)
                    logging.info(f"Seed added to frontier: {url}") # New: confirm seed added
                else:
                    logging.info(f"Skipping seed URL {url} due to robots.txt rules.")

        logging.info(f"[START CRAWL] Frontier initialized with {len(self.frontier)} URLs.")
        logging.info(f"Max depth: {self.max_depth}, Default delay: {self.default_delay}s, SimHash threshold: {self.simhash_threshold}")
       
        # Track some statistics
        crawled_count = 0
        skipped_robots = 0
        skipped_duplicate_content = 0
        skipped_non_english = 0
        skipped_already_processed = 0 # New counter

        while self.frontier:
            priority, url, depth = heapq.heappop(self.frontier)

            # Check if this URL has been *processed* before (based on doc_id)
            if url in self.url_to_doc_id:
                logging.info(f"Skipping {url} (already processed and indexed in a previous run).")
                skipped_already_processed += 1
                continue

            if depth > self.max_depth:
                logging.info(f"Skipping {url} (max depth {self.max_depth} reached).")
                continue

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            rp = self._get_robot_parser(url)
            if not rp.can_fetch(self.user_agent, url):
                logging.info(f"[ROBOTS] Skipping {url} due to robots.txt rules for {domain}.")
                skipped_robots += 1
                continue

            required_delay = self.domain_delays.get(domain, self.default_delay)
            
            if domain in self.last_fetch_time:
                time_since_last_fetch = time.time() - self.last_fetch_time[domain]
                if time_since_last_fetch < required_delay:
                    sleep_duration = required_delay - time_since_last_fetch
                    logging.info(f"[DELAY] Sleeping for {sleep_duration:.2f}s for {domain} to respect crawl-delay.")
                    time.sleep(sleep_duration)
            
            self.last_fetch_time[domain] = time.time() 

            # NEW: Start timer for fetch and process speed
            start_page_time = time.time()

            logging.info(f"\n[CRAWL] Fetching: {url} (Depth: {depth}, Priority: {-priority:.2f})") # Show priority
            try:
                headers = {'User-Agent': self.user_agent}
                resp = requests.get(url, headers=headers, timeout=10) # Increased timeout for robustness
                resp.raise_for_status()
                html = resp.text
            except requests.exceptions.RequestException as e:
                logging.error(f"fetching {url}: {e}")
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            
            cleaned_text_for_simhash = self._get_cleaned_text_for_simhash(soup)
            if cleaned_text_for_simhash:
                current_page_simhash = self._compute_simhash(cleaned_text_for_simhash)
            else:
                current_page_simhash = 0

            is_content_duplicate = False
            for existing_simhash in self.seen_simhashes:
                if self._hamming_distance(current_page_simhash, existing_simhash) <= self.simhash_threshold:
                    logging.info(f"[SIMHASH] Skipping {url} (content duplicate with existing hash).")
                    is_content_duplicate = True
                    skipped_duplicate_content += 1
                    break

            if is_content_duplicate:
                continue

            self.seen_simhashes.add(current_page_simhash)
            self._save(self.path_to_simhashes, self.seen_simhashes)
            
            tokens_for_indexing = self._preprocess_text(soup)
            if not tokens_for_indexing:
                logging.info(f"[LANG] Skipping {url} (non-English or no extractable text).")
                skipped_non_english += 1
                continue

            doc_id = self._get_id(url)
            self._save_crawled(doc_id, {'url': url, 'tokens': tokens_for_indexing})
            self.url_to_doc_id[url] = doc_id
            crawled_count += 1

            # NEW: End timer and print processing speed
            end_page_time = time.time()
            processing_time = end_page_time - start_page_time
            logging.info(f"[SUCCESS] Processed {url} (Doc ID: {doc_id}) in {processing_time:.2f} seconds.")

            if depth < self.max_depth:
                current_page_text_for_scoring = self._get_cleaned_text_for_simhash(soup)
                current_page_tokens_for_scoring = self._clean_for_scoring(current_page_text_for_scoring) 
                
                links_added_from_page = 0
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    href, _ = urldefrag(href)
                    p = urlparse(href)
                    
                    if p.scheme in ("http", "https") and href not in self.visited_urls_in_queue:
                        target_rp = self._get_robot_parser(href)
                        if target_rp.can_fetch(self.user_agent, href):
                            anchor_text = a.get_text(strip=True)
                            
                            new_priority = self._calculate_priority(
                                href,
                                anchor_text,
                                current_page_tokens_for_scoring,
                                depth + 1
                            )
                            
                            heapq.heappush(self.frontier, (new_priority, href, depth + 1))
                            self.visited_urls_in_queue.add(href)
                            links_added_from_page += 1
                        # else:
                            # print(f"  [LINK_SKIP] Extracted link {href} skipped by robots.txt.") # Too verbose, uncomment for deep debugging
                logging.info(f"[LINKS] Added {links_added_from_page} new links to frontier. Frontier size: {len(self.frontier)}.")
            else:
                logging.info(f"[LINKS] Max depth reached, no new links added from {url}.")

        logging.info(f"\n[CRAWL COMPLETE] Finished crawling.")
        logging.info(f"Total pages crawled and indexed: {crawled_count}")
        logging.info(f"Skipped by robots.txt: {skipped_robots}")
        logging.info(f"Skipped as content duplicates (SimHash): {skipped_duplicate_content}")
        logging.info(f"Skipped as non-English or no text: {skipped_non_english}")
        logging.info(f"Skipped (already processed in previous run): {skipped_already_processed}")
        logging.info(f"Remaining URLs in frontier (not crawled): {len(self.frontier)}")


    # NEW METHOD: _get_robot_parser
    def _get_robot_parser(self, url):
        """
        Retrieves or creates a RobotFileParser for a given URL's domain.
        Caches parsers to avoid re-fetching robots.txt for the same domain.
        Handles fetching robots.txt for the first time and determining crawl delay.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain not in self.robot_parsers:
            robot_url = urlunparse(parsed_url._replace(path="/robots.txt", params="", query="", fragment=""))
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robot_url)

            current_domain_delay = self.default_delay # Initialize with default

            try:
                headers = {'User-Agent': self.user_agent}
                response = requests.get(robot_url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    rp.parse(response.text.splitlines())
                    logging.info(f"Successfully fetched and parsed robots.txt for {domain}")
                    
                    # Get crawl delay from robots.txt
                    c_delay = rp.crawl_delay(self.user_agent) 
                    if c_delay is not None:
                        # Use the robots.txt delay directly if specified, to speed up if allowed.
                        current_domain_delay = c_delay 
                        logging.info(f"Applying robots.txt crawl-delay of {c_delay}s for {domain}. Effective delay: {current_domain_delay}s")
                    else:
                        # If no specific crawl-delay from robots.txt, use our default
                        logging.info(f"No specific crawl-delay in robots.txt for {domain}. Using default delay: {self.default_delay}s")
                        current_domain_delay = self.default_delay 
                    
                elif response.status_code in (401, 403):
                    rp.disallow_all = True
                    logging.info(f"Access denied to robots.txt for {domain}. Disallowing all.")
                elif response.status_code >= 400 and response.status_code < 500:
                    rp.allow_all = True
                    logging.info(f"robots.txt not found for {domain} (Status: {response.status_code}). Allowing all.")
                else:
                    rp.allow_all = True
                    logging.info(f"Error fetching robots.txt for {domain} (Status: {response.status_code}). Allowing all.")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to fetch robots.txt for {domain}: {e}. Allowing all.")
                rp.allow_all = True

            self.robot_parsers[domain] = rp
            self.domain_delays[domain] = current_domain_delay # Store the determined delay for this domain
            
            # This sleep is for fetching robots.txt itself, not the content pages.
            time.sleep(self.default_delay) 

        return self.robot_parsers[domain]

    # NEW/MODIFIED METHOD: _calculate_priority
    def _calculate_priority(self, url, anchor_text, source_page_tokens, current_depth):
        score = 0
        # Define weights for each relevance tier
        VERY_RELEVANT_WEIGHT = 20
        RELEVANT_WEIGHT = 10
        MODERATELY_RELEVANT_WEIGHT = 5
        DEPTH_PENALTY_WEIGHT = 5

        # 1. Anchor Text Relevance (Highest Impact)
        if anchor_text:
            cleaned_anchor_text_tokens = set(self._clean_for_scoring(anchor_text))
            
            # Check against each tier
            common_vr_keywords = cleaned_anchor_text_tokens.intersection(self.very_relevant_keywords)
            common_r_keywords = cleaned_anchor_text_tokens.intersection(self.relevant_keywords)
            common_mr_keywords = cleaned_anchor_text_tokens.intersection(self.moderately_relevant_keywords)
            
            score += len(common_vr_keywords) * VERY_RELEVANT_WEIGHT
            score += len(common_r_keywords) * RELEVANT_WEIGHT
            score += len(common_mr_keywords) * MODERATELY_RELEVANT_WEIGHT

        # 2. URL Keyword Relevance (High Impact)
        parsed_url = urlparse(url)
        url_components_text = (parsed_url.netloc + parsed_url.path + parsed_url.query).lower()
        url_components_tokens = set(self._clean_for_scoring(url_components_text)) 
        
        common_vr_url = url_components_tokens.intersection(self.very_relevant_keywords)
        common_r_url = url_components_tokens.intersection(self.relevant_keywords)
        common_mr_url = url_components_tokens.intersection(self.moderately_relevant_keywords)
        
        score += len(common_vr_url) * VERY_RELEVANT_WEIGHT
        score += len(common_r_url) * RELEVANT_WEIGHT
        score += len(common_mr_url) * MODERATELY_RELEVANT_WEIGHT

        # 3. Source Page Context Relevance (Medium Impact)
        if source_page_tokens:
            source_tokens_set = set(source_page_tokens)
            
            common_vr_source = source_tokens_set.intersection(self.very_relevant_keywords)
            common_r_source = source_tokens_set.intersection(self.relevant_keywords)
            common_mr_source = source_tokens_set.intersection(self.moderately_relevant_keywords)

            score += len(common_vr_source) * (VERY_RELEVANT_WEIGHT / 2)
            score += len(common_r_source) * (RELEVANT_WEIGHT / 2)
            score += len(common_mr_source) * (MODERATELY_RELEVANT_WEIGHT / 2)

        # 4. Depth Penalty
        score -= (current_depth * DEPTH_PENALTY_WEIGHT)

        score = max(0, score)
        return -score
    # NEW/MODIFIED METHOD: _clean_for_scoring
    def _clean_for_scoring(self, text):
        """
        Basic cleaning: lowercase, remove punctuation, normalize whitespace, and return tokens.
        This is used for anchor text, URL components, and source page text for relevance scoring.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) # Remove anything that's not alphanumeric or whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split() # Return as a list of tokens

    # NEW/MODIFIED METHOD: _get_cleaned_text_for_simhash
    def _get_cleaned_text_for_simhash(self, soup):
        """
        Extracts and cleans text specifically for SimHash calculation.
        This typically involves stripping more boilerplate elements.
        """
        # Exclude common boilerplate tags for better content focus in SimHash
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ")

        # Lowercase and remove excessive whitespace for SimHash
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # NEW METHOD: _compute_simhash (ensure hashlib is imported)
    def _compute_simhash(self, text, num_bits=64):
        """
        Computes the SimHash of the given text.
        Uses a simple bag-of-words approach with MD5 hashing for features.
        """
        if not text:
            return 0

        features = {}
        for word in text.split():
            feature_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            features[feature_hash] = features.get(feature_hash, 0) + 1

        v = [0] * num_bits
        for feature_hash, weight in features.items():
            for i in range(num_bits):
                if (feature_hash >> i) & 1:
                    v[i] += weight
                else:
                    v[i] -= weight

        simhash_val = 0
        for i in range(num_bits):
            if v[i] >= 0:
                simhash_val |= (1 << i)
        return simhash_val

    # NEW METHOD: _hamming_distance
    def _hamming_distance(self, hash1, hash2):
        """Calculates the Hamming distance between two integers (representing hashes)."""
        x = hash1 ^ hash2
        set_bits = 0
        while x > 0:
            x &= (x - 1)
            set_bits += 1
        return set_bits

    # MODIFIED METHOD: _preprocess_text
    def _preprocess_text(self, soup):
        # Extract visible text (from the original soup, not the one modified for SimHash)
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
            return None # Return None to indicate this page should be skipped for indexing
        
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        filtered_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in stop_words and len(token) > 2
        ]
        return filtered_tokens

    # MODIFIED METHOD: _get_id
    def _get_id(self, url):
        # Use existing ID if URL was previously crawled and saved
        if url in self.url_to_doc_id:
            return self.url_to_doc_id[url]

        # Otherwise, assign a new ID
        if self.crawled_data:
            return max(self.crawled_data.keys()) + 1
        return 0 # First document will be ID 0

    # MODIFIED METHOD: _save_crawled
    def _save_crawled(self, doc_id, doc_info):
        self.crawled_data[doc_id] = doc_info

        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)

    # NEW METHOD: _save (generic save)
    def _save(self, path, data):
        """Generic save method for any pickleable data."""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # TODO these three methods should be unified
    def _save_all(self):
        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)

    # MODIFIED METHOD: _load
    def _load(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            if path == self.path_to_crawled_data:
                logging.warning(f"No previous crawl data found, starting fresh for crawled_data.")
                return {}
            elif path == self.path_to_simhashes:
                logging.warning(f"No previous SimHashes found, starting fresh for SimHash store.")
                return set()
            else:
                raise

    def _handle_interrupt(self, signum, frame):
        logging.info("Interrupted! Saving current progress...")
        self._save_all()
        sys.exit(0)


