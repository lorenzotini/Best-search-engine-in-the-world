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
        self.path_to_frontier = 'data/frontier.pkl'
        self.path_to_visited_urls_in_queue = 'data/visited_urls_in_queue.pkl'

        self.crawled_data = self._load(self.path_to_crawled_data)
        self.seen_simhashes = self._load(self.path_to_simhashes)
        self.frontier = self._load(self.path_to_frontier) # NEW: Load persistent frontier
        self.visited_urls_in_queue = self._load(self.path_to_visited_urls_in_queue) # NEW: Load persistent visited URLs

        signal.signal(signal.SIGINT, self._handle_interrupt) 
        
        # Priority queue for frontier: (priority, url, depth)
        self.frontier = []
        # Keep track of URLs that have been added to the frontier to avoid duplicates
        self.visited_urls_in_queue = set()

        # For mapping URL to doc_id (helpful for debugging and future relevance)
        self.url_to_doc_id = {url_info['url']: doc_id for doc_id, url_info in self.crawled_data.items() if 'url' in url_info}

        # Stores RobotFileParser objects, keyed by domain (netloc)
        self.robot_parsers = {}
        # Stores the effective crawl delay for each domain
        self.domain_delays = {}
        # User-Agent string for your crawler. Be descriptive and include a contact URL.
        self.user_agent = "TübingenSearchBot_UniProject/4.20 (https://alma.uni-tuebingen.de/alma/pages/startFlow.xhtml?_flowId=detailView-flow&unitId=78284&periodId=228&navigationPosition=studiesOffered,searchCourses)" 
            
            #Tiered keywords for weighted priority
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

        # NEW: Integrate new seeds into the existing frontier intelligently
        seeds_already_known = 0
        seeds_added_count = 0
        total_seeds_to_process = len(self.seeds)

        #logging.info(f"Initiating seed processing. Total seeds: {total_seeds_to_process}.")

        for url in self.seeds:
            # MODIFIED: Normalize URL before checks for consistency
            normalized_url, _ = urldefrag(urljoin(url, url))
            parsed_url = urlparse(normalized_url)

            # Skip malformed URLs early
            if not parsed_url.scheme or not parsed_url.netloc:
                logging.warning(f"[SEED_SKIP] Malformed seed URL: '{url}'. Skipping.")
                seeds_already_known += 1
                continue

            # Check if already processed (crawled and indexed)
            if normalized_url in self.url_to_doc_id:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' already processed (Doc ID: {self.url_to_doc_id[normalized_url]}).")
                seeds_already_known += 1
                continue

            # Check if already in frontier or visited_urls_in_queue
            if normalized_url in self.visited_urls_in_queue:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' already in frontier or visited. Skipping.")
                seeds_already_known += 1
                continue

            # Check robots.txt for the seed URL before adding
            rp = self._get_robot_parser(normalized_url)
            if rp.can_fetch(self.user_agent, normalized_url):
                priority = self._calculate_priority(normalized_url, "", [], 0)
                heapq.heappush(self.frontier, (priority, normalized_url, 0))
                self.visited_urls_in_queue.add(normalized_url) # NEW: Add to visited_urls_in_queue when pushed to frontier
                logging.info(f"[SEED_ADD] Added new seed to frontier: '{normalized_url}'.")
                seeds_added_count += 1
            else:
                logging.info(f"[SEED_SKIP] Seed '{normalized_url}' disallowed by robots.txt.")
                seeds_already_known += 1

        logging.info(f"[START CRAWL] Frontier initialized with {len(self.frontier)} URLs.")
        logging.info(f"Max depth: {self.max_depth}, Default delay: {self.default_delay}s, SimHash threshold: {self.simhash_threshold}")
       
        # Track some statistics
        crawled_count = 0
        skipped_robots = 0
        skipped_duplicate_content = 0
        skipped_non_english = 0
        skipped_already_processed = 0 

        while self.frontier:
            priority, url, depth = heapq.heappop(self.frontier)

            # NEW: Check if this URL has been *processed* before (based on doc_id)
            if url in self.url_to_doc_id:
                logging.info(f"[ALREADY_PROCESSED] Skipping {url} (already processed and indexed in a previous run).")
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

            start_page_time = time.time()

            logging.info(f"\n[CRAWL] Fetching: {url} (Depth: {depth}, Priority: {-priority:.2f})")
            try:
                headers = {'User-Agent': self.user_agent}
                resp = requests.get(url, headers=headers, timeout=10)
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

                    if p.scheme in ("http", "https"):
                        # NEW: Check if already processed (crawled and indexed)
                        if href in self.url_to_doc_id:
                            continue
                        # NEW: Check if already in frontier/visited_urls_in_queue
                        if href in self.visited_urls_in_queue:
                            continue

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
                            self.visited_urls_in_queue.add(href) # NEW: Add to visited_urls_in_queue when adding to frontier
                            links_added_from_page += 1
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

        # NEW: Explicitly save frontier and visited_urls_in_queue on normal completion
        self._save(self.path_to_frontier, self.frontier)
        self._save(self.path_to_visited_urls_in_queue, self.visited_urls_in_queue)
        logging.info("Frontier and visited URLs saved on completion.")


    def _get_robot_parser(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain not in self.robot_parsers:
            robot_url = urlunparse(parsed_url._replace(path="/robots.txt", params="", query="", fragment=""))
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robot_url)

            current_domain_delay = self.default_delay

            try:
                headers = {'User-Agent': self.user_agent}
                # The error occurs here or during subsequent redirect resolution
                response = requests.get(robot_url, headers=headers, timeout=5)

                if response.status_code == 200:
                    # Requests *should* have handled encoding for response.text by now,
                    # but if there's still an issue with the _content_ (less likely
                    # with your current traceback, but good practice):
                    try:
                        robot_content = response.text
                    except UnicodeDecodeError:
                        logging.warning(f"UTF-8 decode failed for robots.txt content from {domain}. Trying ISO-8859-1.")
                        try:
                            robot_content = response.content.decode('iso-8859-1')
                        except UnicodeDecodeError:
                            logging.error(f"Failed to decode robots.txt content for {domain} with ISO-8859-1. Allowing all.")
                            rp.allow_all = True
                            self.robot_parsers[domain] = rp
                            self.domain_delays[domain] = current_domain_delay
                            time.sleep(self.default_delay)
                            return rp

                    rp.parse(robot_content.splitlines())
                    logging.info(f"Successfully fetched and parsed robots.txt for {domain}")

                    c_delay = rp.crawl_delay(self.user_agent)
                    if c_delay is not None:
                        current_domain_delay = c_delay
                        logging.info(f"Applying robots.txt crawl-delay of {c_delay}s for {domain}. Effective delay: {current_domain_delay}s")
                    else:
                        logging.info(f"No specific crawl-delay in robots.txt for {domain}. Using default delay: {self.default_delay}s")
                        current_domain_delay = self.default_delay

                elif response.status_code in (401, 403):
                    rp.disallow_all = True
                    logging.info(f"Access denied to robots.txt for {domain}. Disallowing all.")
                elif response.status_code >= 400 and response.status_code < 500:
                    # 4xx errors for robots.txt often mean it doesn't exist, which implies allow all
                    rp.allow_all = True
                    logging.info(f"robots.txt not found for {domain} (Status: {response.status_code}). Allowing all.")
                else:
                    # Other errors (e.g., 5xx server errors, or unexpected 3xx redirects)
                    rp.allow_all = True
                    logging.info(f"Error fetching robots.txt for {domain} (Status: {response.status_code}). Allowing all.")

            # Catch RequestException, which includes UnicodeDecodeError during the request lifecycle
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to fetch or process robots.txt for {domain} due to: {e}. Allowing all for this domain.")
                rp.allow_all = True # Default to allowing all if there's any issue fetching/parsing robots.txt

            self.robot_parsers[domain] = rp
            self.domain_delays[domain] = current_domain_delay
            time.sleep(self.default_delay)

        return self.robot_parsers[domain]

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

    def _clean_for_scoring(self, text):
        """
        Basic cleaning: lowercase, remove punctuation, normalize whitespace, and return tokens.
        This is used for anchor text, URL components, and source page text for relevance scoring.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) # Remove anything that's not alphanumeric or whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split() # Return as a list of tokens


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

    # (ensure hashlib is imported)
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

    # (generic save)
    def _save(self, path, data):
        """Generic save method for any pickleable data."""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # TODO these three methods should be unified
    def _save_all(self):
        with open(self.path_to_crawled_data, "wb") as f:
            pickle.dump(self.crawled_data, f)

    def _load(self, path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logging.info(f"Loaded existing data from {path}") # NEW: Log which path loaded
                return data
            except FileNotFoundError:
                logging.warning(f"No previous data found for {path}. Initializing empty.") # MODIFIED: Unified warning message
                if path == self.path_to_crawled_data:
                    return {}
                elif path == self.path_to_simhashes:
                    return set()
                # NEW: Handle new paths for frontier and visited_urls_in_queue
                elif path == self.path_to_frontier:
                    return [] # Frontier is a list (heapq)
                elif path == self.path_to_visited_urls_in_queue:
                    return set()
                else:
                    raise # If it's an unrecognized path, re-raise the error

    def _handle_interrupt(self, signum, frame):
        logging.info("Interrupted! Saving current progress before exiting...")
        # NEW: Explicitly save frontier and visited_urls_in_queue on interrupt
        self._save(self.path_to_frontier, self.frontier)
        self._save(self.path_to_visited_urls_in_queue, self.visited_urls_in_queue)
        # Call original _save for crawled_data and simhashes to ensure they're saved
        self._save(self.path_to_crawled_data, self.crawled_data)
        self._save(self.path_to_simhashes, self.seen_simhashes)
        logging.info("Frontier and visited URLs saved on interrupt.")
        sys.exit(0)

if __name__ == "__main__":
  
    seeds = ["https://visit-tubingen.co.uk/welcome-to-tubingen/",
             "https://www.tuebingen.de/",
             "https://www.tuebingen.de/#",
             "https://uni-tuebingen.de/en/",
             "https://en.wikipedia.org/wiki/T%C3%BCbingen",
             "https://www.germany.travel/en/cities-culture/tuebingen.html",
             "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html"
             ]

    # Initialize the crawler
    # max_depth: How many layers deep the crawler will go. Start with 1 or 2 for testing.
    # delay: Delay between requests to the same domain (in seconds). Adjust as needed.
    # simhash_threshold: How similar content can be before being considered a duplicate.
    crawler = OfflineCrawler(
        seeds=seeds,
        max_depth=2,  # Start with a low depth for quick tests
        delay=0.5,      # Be polite with the crawl delay
        simhash_threshold=3
    )

    # Run the crawler
    crawler.run()

    print("\nCrawler finished.")

