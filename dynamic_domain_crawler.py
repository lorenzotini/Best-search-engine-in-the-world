import os
import time
import threading
import queue
import pickle
import requests
import re
import hashlib
import urllib.robotparser

from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs

# Ensure NLTK data is present
nltk.download('stopwords', quiet=True)
nltk.download('punkt',    quiet=True)
nltk.download('wordnet',  quiet=True)

class DynamicDomainCrawler:
    SAVE_INTERVAL = 50     # batch save every 50 new pages
    IDLE_TIMEOUT = 5       # seconds before a domain-worker exits on empty queue

    def __init__(self,
                 seeds,
                 max_depth=2,
                 simhash_threshold=3,
                 user_agent=None,
                 max_threads=100,
                 time_limit=None,        # minutes
                 max_new_pages=None):    # max new/updated pages to fetch
        # Configuration
        self.seeds           = seeds
        self.max_depth       = max_depth
        self.simhash_thresh  = simhash_threshold
        self.user_agent      = user_agent or "TuebingenSearchBot/1.0"
        self.default_delay   = 0.5
        self.max_threads     = max_threads
        self.time_limit      = time_limit
        self.max_new_pages   = max_new_pages

        # Statistics
        self.start_time         = None
        self.urls_visited_count = 0    # total fetch attempts
        self.new_pages_count    = 0    # newly added or updated pages

        # Load persisted crawl data
        self.crawled_data = self._load('crawled_data.pkl', {})
        for doc_id, info in self.crawled_data.items():
            info.setdefault('simhash', None)
            info.setdefault('last_modified', None)
        self.seen_simhashes = self._load('simhashes.pkl', set())
        self.url_to_doc_id  = {info['url']: doc_id for doc_id, info in self.crawled_data.items()}

        # Frontier queues per domain
        self.domain_queues  = defaultdict(queue.Queue)
        self.active_threads = {}          # domain -> Thread
        self.domain_lock    = threading.Lock()

        # Visited URLs set to avoid duplicates in frontier
        self.visited_urls   = set(self.url_to_doc_id)
        self.visited_lock   = threading.Lock()

        # Robots.txt caches
        self.robot_parsers  = {}
        self.domain_delays  = {}
        self.robot_lock     = threading.Lock()
        self.last_fetch     = {}

        # Data lock for crawled_data and simhashes
        self.data_lock = threading.Lock()

        # NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Ranking keyword sets (example)
        self.very_relevant = {"tuebingen", "university", "neckar"}
        self.relevant      = {"tourism", "restaurants", "culture"}
        self.moderate      = {"news", "events", "info"}

        # Load or initialize frontier
        frontier_map = self._load('frontier.pkl', None)
        if frontier_map is not None:
            total = 0
            for dom, lst in frontier_map.items():
                for url, depth in lst:
                    self.domain_queues[dom].put((url, depth))
                    self.visited_urls.add(url)
                    total += 1
            print(f"[INFO] Restored frontier with {total} URLs")

        # Always re-queue seeds and existing URLs to check for updates
        existing = self._frontier_urls()
        # 1) Seeds
        for url in self.seeds:
            if url not in existing:
                dom = urlparse(url).netloc
                if self._get_robot_parser(dom).can_fetch(self.user_agent, url):
                    self.domain_queues[dom].put((url, 0))
                    print(f"[INFO] Seed re-queued: {url}")
        # 2) Existing crawled URLs
        for url in list(self.url_to_doc_id):
            if url not in existing:
                dom = urlparse(url).netloc
                self.domain_queues[dom].put((url, 0))
        print(f"[INFO] {len(self.url_to_doc_id)} existing URLs re-queued for update")

    def _frontier_urls(self):
        """Return set of URLs currently in frontier queues."""
        s = set()
        for q in self.domain_queues.values():
            with q.mutex:
                s |= {item[0] for item in q.queue}
        return s

    def _load(self, path, default):
        if default is None and not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return default

    def _save(self, path, obj):
        with self.data_lock:
            with open(path,'wb') as f:
                pickle.dump(obj, f)

    def _get_robot_parser(self, domain):
        with self.robot_lock:
            if domain in self.robot_parsers:
                return self.robot_parsers[domain]
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f'https://{domain}/robots.txt')
            delay = self.default_delay
            try:
                resp = requests.get(rp.url, headers={'User-Agent': self.user_agent}, timeout=5)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    cd = rp.crawl_delay(self.user_agent)
                    if cd is not None:
                        delay = cd
                        print(f"[INFO] crawl-delay={delay}s for {domain}")
                elif resp.status_code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True
            except Exception as e:
                rp.allow_all = True
                print(f"[ERROR] fetching robots.txt for {domain}: {e}")
            self.robot_parsers[domain] = rp
            self.domain_delays[domain] = delay
            return rp

    def _clean(self, text):
        t = re.sub(r'[^\w\s]', ' ', text.lower())
        return re.sub(r'\s+', ' ', t).split()

    def _priority(self, url, anchor, toks, depth):
        score = 0
        a = set(self._clean(anchor))
        score += len(a & self.very_relevant) * 20
        score += len(a & self.relevant) * 10
        score += len(a & self.moderate) * 5
        u = set(self._clean(url))
        score += len(u & self.very_relevant) * 20
        score += len(u & self.relevant) * 10
        score += len(u & self.moderate) * 5
        if toks:
            s = set(toks)
            score += len(s & self.very_relevant) * 10
            score += len(s & self.relevant) * 5
            score += len(s & self.moderate) * 2
        score -= depth * 5
        return -max(score, 0)

    def _simhash(self, text):
        freq = {}
        for w in text.split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            freq[h] = freq.get(h, 0) + 1
        vec = [0] * 64
        for h, w in freq.items():
            for i in range(64):
                vec[i] += w if ((h >> i) & 1) else -w
        hval = 0
        for i, v in enumerate(vec):
            if v >= 0:
                hval |= (1 << i)
        return hval

    def _pre(self, text):
        try:
            langs = detect_langs(text)
        except:
            return None
        if not any(l.lang == 'en' and l.prob >= 0.9 for l in langs):
            return None
        txt = re.sub(r'[^a-z\s]', ' ', text.lower())
        toks = word_tokenize(txt)
        return [self.lemmatizer.lemmatize(t) for t in toks
                if t not in self.stop_words and len(t) > 2]

    def _extract(self, soup, url, depth):
        ctoks = self._clean(soup.get_text(' '))
        out = []
        for a in soup.find_all('a', href=True):
            href, _ = urldefrag(urljoin(url, a['href']))
            dom = urlparse(href).netloc
            if href.startswith(('http://', 'https://')):
                pr = self._priority(href, a.get_text(strip=True), ctoks, depth + 1)
                out.append((dom, href, depth + 1, pr))
        return out

    def _process(self, url, depth):
        dom = urlparse(url).netloc
        rp = self._get_robot_parser(dom)
        if not rp.can_fetch(self.user_agent, url):
            return []
        last = self.last_fetch.get(dom, 0)
        wait = self.domain_delays.get(dom, self.default_delay) - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        self.last_fetch[dom] = time.time()

        try:
            resp = requests.get(url, headers={'User-Agent': self.user_agent}, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] fetch {url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        txt = soup.get_text(' ')
        new_sim = self._simhash(txt)
        last_mod = resp.headers.get('Last-Modified')
        updated = False

        with self.data_lock:
            if url in self.url_to_doc_id:
                doc = self.crawled_data[self.url_to_doc_id[url]]
                if last_mod and doc.get('last_modified') == last_mod:
                    print(f"[UP-TO-DATE] {url}")
                elif doc.get('simhash') == new_sim:
                    print(f"[UP-TO-DATE-SIM] {url}")
                else:
                    print(f"[UPDATE] {url}")
                    self.seen_simhashes.discard(doc.get('simhash'))
                    doc['simhash']       = new_sim
                    doc['last_modified'] = last_mod
                    doc['tokens']        = self._pre(txt) or []
                    self.new_pages_count += 1
                    updated = True
            else:
                dup = any(bin(new_sim ^ s).count('1') <= self.simhash_thresh
                          for s in self.seen_simhashes)
                if dup:
                    print(f"[DUP] {url}")
                else:
                    toks = self._pre(txt)
                    if toks:
                        did = max(self.crawled_data.keys(), default=-1) + 1
                        self.crawled_data[did] = {
                            'url': url,
                            'tokens': toks,
                            'simhash': new_sim,
                            'last_modified': last_mod
                        }
                        self.url_to_doc_id[url] = did
                        self.seen_simhashes.add(new_sim)
                        self.new_pages_count += 1
                        print(f"[NEW] {url} -> doc{did}")

            # batch save
            if self.new_pages_count > 0 and self.new_pages_count % self.SAVE_INTERVAL == 0:
                self._save('simhashes.pkl', self.seen_simhashes)
                self._save('crawled_data.pkl', self.crawled_data)

        return self._extract(soup, url, depth) if updated else []

    def _domain_worker(self, domain):
        print(f"[THREAD-START] {domain}")
        q = self.domain_queues[domain]
        while not self.stop_event.is_set():
            try:
                url, depth = q.get(timeout=self.IDLE_TIMEOUT)
            except queue.Empty:
                break

            # increment fetch counter
            with self.stats_lock:
                self.urls_visited_count += 1

            # print stats
            active      = len(self.active_threads)
            visited_all = self.urls_visited_count
            new_pages   = self.new_pages_count
            total_uni   = len(self.visited_urls)
            print(f"[STATS] ActiveThreads={active}/{self.max_threads} | "
                  f"VisitedURLs={visited_all} | NewPages={new_pages} | TotalUniqueURLs={total_uni}")

            # check limits
            elapsed = (time.time() - self.start_time) / 60
            with self.stats_lock:
                if (self.time_limit and elapsed >= self.time_limit) or \
                   (self.max_new_pages and self.new_pages_count >= self.max_new_pages):
                    self.stop_event.set()

            links = self._process(url, depth)
            for dom2, href, d2, prio in links:
                with self.visited_lock:
                    if href in self.visited_urls:
                        continue
                    self.visited_urls.add(href)
                with self.domain_lock:
                    self.domain_queues[dom2].put((href, d2))
            q.task_done()

        print(f"[THREAD-END] {domain}")
        with self.domain_lock:
            del self.active_threads[domain]

    def _manager(self):
        print("[MANAGER] starting")
        while not self.stop_event.is_set():
            with self.domain_lock:
                for dom, q in list(self.domain_queues.items()):
                    if (dom not in self.active_threads
                        and not q.empty()
                        and len(self.active_threads) < self.max_threads):
                        t = threading.Thread(target=self._domain_worker,
                                             args=(dom,), daemon=True)
                        self.active_threads[dom] = t
                        t.start()
            time.sleep(0.5)
        print("[MANAGER] stopping")

    def run(self):
        self.start_time = time.time()
        self.stop_event  = threading.Event()
        self.stats_lock  = threading.Lock()

        def on_exit():
            fmap = {dom: list(q.queue) for dom, q in self.domain_queues.items()}
            self._save('frontier.pkl', fmap)
            self._save('simhashes.pkl', self.seen_simhashes)
            self._save('crawled_data.pkl', self.crawled_data)

        mgr = threading.Thread(target=self._manager, daemon=True)
        mgr.start()

        # Warte bis stop_event gesetzt wird
        while not self.stop_event.is_set():
            time.sleep(1)

        # Warte, bis alle Domain-Worker-Threads wirklich beendet sind
        with self.domain_lock:
            threads = list(self.active_threads.values())
        for t in threads:
            t.join()

        # Jetzt erst sicher speichern
        on_exit()

        # final summary
        active      = len(self.active_threads)
        visited_all = self.urls_visited_count
        new_pages   = self.new_pages_count
        total_uni   = len(self.visited_urls)
        left        = sum(q.qsize() for q in self.domain_queues.values())

        print("\n[Crawl Complete]")
        print(f"  ActiveThreads     {active}/{self.max_threads}")
        print(f"  VisitedURLs       {visited_all}")
        print(f"  NewPagesAdded     {new_pages}")
        print(f"  TotalUniqueURLs   {total_uni}")
        print(f"  FrontierRemaining {left}")

if __name__ == '__main__':
    seeds = [
        'https://www.germany.travel/en/cities-culture/tuebingen.html',
        'https://uni-tuebingen.de',
        'https://de.wikipedia.org/wiki/Tübingen'
    ]
    crawler = DynamicDomainCrawler(
        seeds,
        max_depth=5,
        simhash_threshold=3,
        max_threads=100,
        time_limit=5,
        max_new_pages=100
    )
    crawler.run()
