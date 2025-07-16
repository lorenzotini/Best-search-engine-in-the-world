import time
import threading
import requests
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from bs4 import BeautifulSoup
import pickle
import re
import hashlib
import urllib.robotparser
from collections import defaultdict
import heapq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs

class OfflineCrawler:
    def __init__(self, seeds, max_depth=2, simhash_threshold=3, user_agent=None):
        # Configuration
        self.max_depth = max_depth
        self.simhash_threshold = simhash_threshold
        self.user_agent = user_agent or (
            "TuebingenSearchBot_UniProject/4.20 "
            "(https://alma.uni-tuebingen.de/alma/pages/startFlow.xhtml?_flowId=detailView-flow&unitId=78284)"
        )
        self.default_delay = 0.5

        # State
        self.seeds = seeds
        # Per-domain priority queues: (priority, url, depth)
        self.domain_queues = defaultdict(list)
        self.visited_urls_in_queue = set()

        # Robots.txt parsers and delays
        self.robot_parsers = {}
        self.domain_delays = {}
        self.last_fetch = {}

        # Storage
        self.seen_simhashes = self._load('simhashes.pkl', set())
        self.crawled_data = self._load('crawled_data.pkl', {})
        # Ensure stored entries have simhash and last_modified fields
        for doc_id, info in self.crawled_data.items():
            info.setdefault('simhash', None)
            info.setdefault('last_modified', None)
        self.url_to_doc_id = {info['url']: doc_id for doc_id, info in self.crawled_data.items()}

        # NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Target keywords for ranking (original list)
        self.target_keywords = {
            # Core Geographic & Identity Keywords
            "tuebingen", "tubingen", "university", "universität", "eberhard karls universität",
            "neckar", "altstadt", "hohentuebingen", "hohentübingen", "baden-wuerttemberg",
            "baden-württemberg", "schwaebische alb", "schwübische alb", "schoenbuch", "schönbuch",

            # General Town Life & Categories
            "city", "town", "municipality", "community", "kreis tuebingen", "information",
            "news", "events", "tourism", "travel", "visitors", "sights", "attractions",
            "living", "citizens", "local", "region", "regional",

            # Education & Research
            "study", "students", "research", "science", "university hospital", "uniklinik",
            "excellence university", "institute", "faculty", "cyber valley",

            # Culture & Arts
            "culture", "art", "theater", "theatre", "museums", "exhibitions", "music",
            "cinema", "books", "literature", "galleries",

            # Events & Festivals (Keep specific German names as they are unique identifiers)
            "stocherkahnrennen", "chocolart", "christmas market", "weihnachtsmarkt",
            "city festival", "stadtfest", "umbrisch-provenzalischer markt",
            "jazz & klassik tage", "sommernachtskino", "fasnet", "karneval", "festivals",

            # Economy & Business
            "companies", "businesses", "jobs", "work", "economy", "trade", "commerce",
            "retail", "startups", "industry", "services",

            # Food & Gastronomy
            "restaurants", "cafes", "gastronomy", "food", "drinks", "bars", "bakery",
            "butcher", "market", "wochenmarkt",

            # Accommodation
            "hotels", "accommodation", "holiday apartments", "guesthouses", "youth hostel",

            # Daily Life & Services
            "housing", "living", "real estate", "traffic", "public transport", "parking",
            "health", "doctors", "pharmacies", "schools", "kindergartens", "sport",
            "clubs", "associations", "town hall", "administration", "waste", "building",

            # Surrounding Towns & Villages (within Tübingen County/Region)
            "reutlingen", "rottenburg am neckar", "moessingen", "mössingen",
            "kirchentellinsfurt", "ammerbuch", "dusslingen", "dußlingen", "gomaringen",
            "kusterdingen", "ofterdingen", "nehren", "dettenhausen"
        }

        # Seed initialization with ranking
        for url in self.seeds:
            domain = urlparse(url).netloc
            rp = self._get_robot_parser(domain)
            if rp.can_fetch(self.user_agent, url):
                priority = self._calculate_priority(url, "", [], 0)
                heapq.heappush(self.domain_queues[domain], (priority, url, 0))
                self.visited_urls_in_queue.add(url)
                print(f"[INFO] Seed added to frontier: {url} (Priority: {-priority})")
            else:
                print(f"[INFO] Skipping seed URL {url} due to robots.txt rules.")
        print(f"\n[START CRAWL] Initialized with {sum(len(q) for q in self.domain_queues.values())} URLs across {len(self.domain_queues)} domains.")
        print(f"Max depth: {self.max_depth}, Default delay: {self.default_delay}s, SimHash threshold: {self.simhash_threshold}\n")

    def _load(self, path, default):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return default

    def _save(self, path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _get_robot_parser(self, domain):
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        robot_url = urlunparse(('https', domain, '/robots.txt', '', '', ''))
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robot_url)
        delay = self.default_delay
        try:
            resp = requests.get(robot_url, headers={'User-Agent': self.user_agent}, timeout=5)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                c = rp.crawl_delay(self.user_agent)
                if c is not None:
                    delay = c
                    print(f"Applying robots.txt crawl-delay {delay}s for {domain}.")
                else:
                    print(f"No specific crawl-delay for {domain}. Using default {self.default_delay}s.")
            elif resp.status_code in (401,403):
                rp.disallow_all = True
                print(f"Access denied to robots.txt for {domain}. Disallowing all.")
            else:
                rp.allow_all = True
                print(f"robots.txt not found or error {resp.status_code} for {domain}. Allowing all.")
        except requests.RequestException as e:
            rp.allow_all = True
            print(f"Failed to fetch robots.txt for {domain}: {e}. Allowing all.")
        self.domain_delays[domain] = delay
        self.robot_parsers[domain] = rp
        time.sleep(self.default_delay)
        return rp

    def _compute_simhash(self, text, bits=64):
        if not text:
            return 0
        freq = {}
        for word in text.split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            freq[h] = freq.get(h, 0) + 1
        v = [0] * bits
        for h, w in freq.items():
            for i in range(bits):
                v[i] += w if ((h >> i) & 1) else -w
        hval = 0
        for i, val in enumerate(v):
            if val >= 0:
                hval |= (1 << i)
        return hval

    def _is_duplicate(self, simhash):
        for s in self.seen_simhashes:
            if bin(simhash ^ s).count('1') <= self.simhash_threshold:
                return True
        return False

    def _preprocess(self, text):
        try:
            langs = detect_langs(text)
        except:
            return None
        if not any(l.lang == 'en' and l.prob >= 0.9 for l in langs):
            return None
        txt = re.sub(r'[^a-z\s]', ' ', text.lower())
        tokens = word_tokenize(txt)
        return [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 2]

    def _calculate_priority(self, url, anchor_text, source_tokens, depth):
        score = 0
        if anchor_text:
            anchor_tokens = set(self._clean_for_scoring(anchor_text))
            score += len(anchor_tokens.intersection(self.target_keywords)) * 10
        pu = urlparse(url)
        comps = (pu.netloc + pu.path + pu.query).lower()
        url_tokens = set(self._clean_for_scoring(comps))
        score += len(url_tokens.intersection(self.target_keywords)) * 7
        if source_tokens:
            src = set(source_tokens)
            score += len(src.intersection(self.target_keywords)) * 3
        score -= depth * 5
        return -max(0, score)

    def _clean_for_scoring(self, text):
        t = text.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t.split()

    def _get_cleaned_text_for_simhash(self, soup):
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            tag.decompose()
        text = soup.get_text(' ')
        return re.sub(r'\s+', ' ', text.lower()).strip()

    def _fetch(self, url, depth, domain):
        rp = self._get_robot_parser(domain)
        if not rp.can_fetch(self.user_agent, url):
            print(f"[ROBOTS] Skipping {url} due to robots.txt for {domain}.")
            return None, None
        last = self.last_fetch.get(domain, 0)
        wait = self.domain_delays.get(domain, self.default_delay) - (time.time() - last)
        if wait > 0:
            print(f"[DELAY] Sleeping {wait:.2f}s for {domain}.")
            time.sleep(wait)
        self.last_fetch[domain] = time.time()
        try:
            resp = requests.get(url, headers={'User-Agent': self.user_agent}, timeout=10)
            resp.raise_for_status()
            return resp.text, resp
        except requests.RequestException as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            return None, None

    def _crawl_domain(self, domain):
        queue = self.domain_queues[domain]
        crawled = skipped_r = skipped_d = skipped_l = updated = 0
        while queue:
            priority, url, depth = heapq.heappop(queue)
            print(f"\n[CRAWL] {domain} fetching {url} (Depth {depth}, Priority {-priority})")
            if depth > self.max_depth:
                print(f"[INFO] Skipping {url} (max depth reached).")
                continue
            html, resp = self._fetch(url, depth, domain)
            if not html:
                skipped_r += 1
                continue
            soup = BeautifulSoup(html, 'html.parser')
            cleaned_text = self._get_cleaned_text_for_simhash(soup)
            new_simhash = self._compute_simhash(cleaned_text)

            if url in self.url_to_doc_id:
                # Existing URL: check for updates
                doc_id = self.url_to_doc_id[url]
                entry = self.crawled_data[doc_id]
                last_mod = resp.headers.get('Last-Modified')
                if last_mod and entry.get('last_modified') == last_mod:
                    print(f"[UP-TO-DATE] {url} (Last-Modified unchanged).")
                    continue
                old_sim = entry.get('simhash')
                if old_sim is not None and new_simhash == old_sim:
                    print(f"[UP-TO-DATE] {url} (simhash unchanged).")
                    if last_mod:
                        entry['last_modified'] = last_mod
                        self._save('crawled_data.pkl', self.crawled_data)
                    continue
                # Content changed: update entry
                print(f"[UPDATE] {url} content changed. Updating.")
                if old_sim is not None:
                    self.seen_simhashes.discard(old_sim)
                self.seen_simhashes.add(new_simhash)
                tokens = self._preprocess(soup.get_text(' ')) or []
                entry['tokens'] = tokens
                entry['simhash'] = new_simhash
                if last_mod:
                    entry['last_modified'] = last_mod
                self._save('crawled_data.pkl', self.crawled_data)
                self._save('simhashes.pkl', self.seen_simhashes)
                updated += 1
                continue_links = True
            else:
                # New URL: standard processing
                if self._is_duplicate(new_simhash):
                    print(f"[SIMHASH] Skipping {url} (duplicate content).")
                    skipped_d += 1
                    continue
                tokens = self._preprocess(soup.get_text(' '))
                if not tokens:
                    print(f"[LANG] Skipping {url} (non-English or empty).")
                    skipped_l += 1
                    continue
                doc_id = max(self.crawled_data.keys(), default=-1) + 1
                self.crawled_data[doc_id] = {
                    'url': url,
                    'tokens': tokens,
                    'simhash': new_simhash,
                    'last_modified': resp.headers.get('Last-Modified')
                }
                self.url_to_doc_id[url] = doc_id
                self.seen_simhashes.add(new_simhash)
                self._save('crawled_data.pkl', self.crawled_data)
                self._save('simhashes.pkl', self.seen_simhashes)
                crawled += 1
                print(f"[SUCCESS] Processed {url} (Doc ID: {doc_id}).")
                continue_links = True

            # Link extraction for new or updated pages
            if continue_links and depth < self.max_depth:
                clean_for_score = self._clean_for_scoring(cleaned_text)
                links_added = 0
                for a in soup.find_all('a', href=True):
                    href, _ = urldefrag(urljoin(url, a['href']))
                    pu = urlparse(href)
                    if pu.scheme in ('http', 'https') and href not in self.visited_urls_in_queue:
                        trp = self._get_robot_parser(pu.netloc)
                        if trp.can_fetch(self.user_agent, href):
                            anchor = a.get_text(strip=True)
                            prio = self._calculate_priority(href, anchor, clean_for_score, depth + 1)
                            heapq.heappush(queue, (prio, href, depth + 1))
                            self.visited_urls_in_queue.add(href)
                            links_added += 1
                print(f"  [LINKS] Added {links_added} new links. Frontier size: {len(queue)}.")

        print(f"[DOMAIN COMPLETE] {domain}: crawled={crawled}, updated={updated}, skipped_robots={skipped_r}, skipped_dup={skipped_d}, skipped_lang={skipped_l}")

    def run(self):
        threads = []
        for domain in self.domain_queues:
            t = threading.Thread(target=self._crawl_domain, args=(domain,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("\n[CRAWL COMPLETE] All domains finished.")

if __name__ == '__main__':
    seeds = ['https://www.germany.travel/en/cities-culture/tuebingen.html']
    crawler = OfflineCrawler(seeds, max_depth=1)
    crawler.run()
