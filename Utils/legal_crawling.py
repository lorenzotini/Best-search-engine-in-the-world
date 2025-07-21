import os
import sys
import time
import requests
import pickle
import logging
import signal
import heapq
import re
import urllib.robotparser

from urllib.parse import urljoin, urldefrag, urlparse, urlunparse
from collections import defaultdict
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect_langs
import hashlib

# NLTK-Daten sicherstellen
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("log/crawler.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class OfflineCrawler:
    def __init__(self, seeds, max_depth=2, delay=0.5, simhash_threshold=3):
        self.seeds = seeds
        self.max_depth = max_depth
        self.default_delay = delay
        self.simhash_threshold = simhash_threshold

        # Pickle-Pfade (unverändert, um alten Dumps kompatibel zu halten)
        self.path_to_crawled_data = 'data/crawled_data.pkl'
        self.path_to_simhashes    = 'data/simhashes.pkl'
        self.path_to_frontier     = 'data/frontier.pkl'
        self.path_to_visited      = 'data/visited_urls_in_queue.pkl'

        # Lade Persistenz
        self.crawled_data             = self._load(self.path_to_crawled_data, {})
        self.seen_simhashes           = self._load(self.path_to_simhashes, set())
        self.frontier                 = self._load(self.path_to_frontier, [])
        self.visited_urls_in_queue    = self._load(self.path_to_visited, set())

        # Mapping URL → doc_id (um Updates zu erkennen)
        self.url_to_doc_id = {
            info['url']: doc_id
            for doc_id, info in self.crawled_data.items()
        }

        # Robots.txt
        self.robot_parsers = {}
        self.domain_delays = {}

        self.user_agent = (
            "TübingenSearchBot_UniProject/4.20 "
            "(kontakt@uni-tuebingen.de)"
        )

        # CTRL-C sauber abfangen
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _load(self, path, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            logging.warning(f"Pickle {path} leer oder korrupt, initialisiere frisch.")
            return default

    def _save(self, path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _save_crawled(self, doc_id, info_dict):
        """Einzeldokument speichern (ohne das gesamte Format zu ändern)."""
        self.crawled_data[doc_id] = info_dict
        self.url_to_doc_id[info_dict['url']] = doc_id
        # Persistiere sofort
        self._save(self.path_to_crawled_data, self.crawled_data)

    def _compute_simhash(self, text, bits=64):
        freqs = {}
        for w in text.split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            freqs[h] = freqs.get(h, 0) + 1
        vec = [0]*bits
        for h, weight in freqs.items():
            for i in range(bits):
                vec[i] += weight if (h>>i)&1 else -weight
        hval = 0
        for i, v in enumerate(vec):
            if v >= 0:
                hval |= 1<<i
        return hval

    def _hamming_distance(self, h1, h2):
        return bin(h1 ^ h2).count('1')

    def _get_robot_parser(self, url):
        domain = urlparse(url).netloc
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f'https://{domain}/robots.txt')
        delay = self.default_delay
        try:
            r = requests.get(rp.url,
                             headers={'User-Agent': self.user_agent},
                             timeout=5)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
                cd = rp.crawl_delay(self.user_agent)
                if cd:
                    delay = cd
                    logging.info(f"[ROBOT] crawl-delay={delay}s for {domain}")
            elif r.status_code in (401,403):
                rp.disallow_all = True
            else:
                rp.allow_all = True
        except Exception as e:
            rp.allow_all = True
            logging.warning(f"robots.txt fetch failed for {domain}: {e}")
        self.robot_parsers[domain] = rp
        self.domain_delays[domain] = delay
        return rp

    def _preprocess_text(self, soup):
        text = soup.get_text(" ")
        # Language check
        try:
            langs = detect_langs(text)
        except:
            return None
        if not any(l.lang=='en' and l.prob>=0.9 for l in langs):
            return None
        txt = re.sub(r'[^a-z\s]', ' ', text.lower())
        toks = word_tokenize(txt)
        return [
            lemmatizer.lemmatize(t)
            for t in toks
            if t not in stop_words and len(t)>2
        ]

    def _handle_interrupt(self, signum, frame):
        logging.info("[INTERRUPT] Abbruchsignal empfangen. Speichere und beende.")
        self._save(self.path_to_frontier, self.frontier)
        self._save(self.path_to_visited, self.visited_urls_in_queue)
        sys.exit(0)

    def run(self):
        """Main-Crawl-Schleife mit Last-Modified- und SimHash-Update-Check."""
        logging.info(f"[START] Seeds: {self.seeds}")
        # Frontier initialisieren, falls leer
        if not self.frontier:
            for url in self.seeds:
                norm = url.strip()
                if norm not in self.visited_urls_in_queue:
                    heapq.heappush(self.frontier, (0, norm, 0))
                    self.visited_urls_in_queue.add(norm)
            logging.info(f"[INIT] Frontier mit {len(self.frontier)} Seeds aufgefüllt.")

        crawled = 0
        while self.frontier:
            priority, url, depth = heapq.heappop(self.frontier)
            # Respektiere Robots
            rp = self._get_robot_parser(url)
            if not rp.can_fetch(self.user_agent, url):
                logging.info(f"[ROBOTS] Skip {url}")
                continue

            # Abbruch, falls tiefer als erlaubt
            if depth > self.max_depth:
                logging.info(f"[DEPTH] Skip {url} (Depth>{self.max_depth})")
                continue

            logging.info(f"[CRAWL] {url} (Depth {depth})")
            # Verzögerung einhalten
            dom = urlparse(url).netloc
            prev = getattr(self, 'last_fetch', {}).get(dom, 0)
            wait = self.domain_delays.get(dom, self.default_delay) - (time.time()-prev)
            if wait>0:
                time.sleep(wait)
            self.last_fetch = {**getattr(self, 'last_fetch', {}), dom: time.time()}

            # Fetch mit HEAD + GET
            try:
                # HEAD für Last-Modified
                head = requests.head(url,
                                     headers={'User-Agent': self.user_agent},
                                     timeout=10,
                                     allow_redirects=True)
                last_mod = head.headers.get('Last-Modified')
                # Entweder 304 oder GET
                get = requests.get(url,
                                   headers={'User-Agent': self.user_agent},
                                   timeout=10)
                get.raise_for_status()
            except Exception as e:
                logging.error(f"[ERROR] fetch {url}: {e}")
                continue

            html = get.text
            soup = BeautifulSoup(html, "html.parser")
            # SimHash der aktuellen Version
            text_for_hash = re.sub(r'\s+', ' ', soup.get_text(" ")).strip()
            cur_sim = self._compute_simhash(text_for_hash)

            # Bereits im Cache?
            if url in self.url_to_doc_id:
                doc_id = self.url_to_doc_id[url]
                old_info = self.crawled_data[doc_id]
                old_mod  = old_info.get('last_modified')
                old_sim  = old_info.get('simhash')

                # 1) Header unverändert?
                if last_mod and old_mod == last_mod:
                    logging.info(f"[UP-TO-DATE] {url} (Last-Modified)")
                    continue

                # 2) SimHash-Distanz zu klein?
                if old_sim is not None and self._hamming_distance(cur_sim, old_sim) <= self.simhash_threshold:
                    logging.info(f"[UP-TO-DATE-SIM] {url} (Hamming-Dist={self._hamming_distance(cur_sim, old_sim)})")
                    # optional: update header-only
                    old_info['last_modified'] = last_mod or old_mod
                    self._save(self.path_to_crawled_data, self.crawled_data)
                    continue

                # 3) Tatsächliches Update
                logging.info(f"[UPDATE] {url} (mod or simhash change)")
            else:
                # völlig neuer Doc
                doc_id = max(self.crawled_data.keys(), default=-1) + 1
                logging.info(f"[NEW] {url} -> doc{doc_id}")

            # Tokens extrahieren
            toks = self._preprocess_text(soup)
            if not toks:
                logging.info(f"[LANG] Skip {url} (non-English)")
                continue

            # Infos speichern (gleiches Pickle-Format!)
            info = {
                'url': url,
                'tokens': toks,
                'simhash': cur_sim,
                'last_modified': last_mod
            }
            self._save_crawled(doc_id, info)
            # Simhash-Pickle updaten
            self.seen_simhashes.add(cur_sim)
            self._save(self.path_to_simhashes, self.seen_simhashes)

            crawled += 1
            # Links nachziehen
            if depth < self.max_depth:
                for a in soup.find_all("a", href=True):
                    href,_ = urldefrag(urljoin(url, a['href']))
                    if href.startswith(("http://","https://")) and href not in self.visited_urls_in_queue:
                        heapq.heappush(self.frontier, (0, href, depth+1))
                        self.visited_urls_in_queue.add(href)
            logging.info(f"[LINKS] Frontier size: {len(self.frontier)}")

        # Ende: Frontier & besucht speichern
        self._save(self.path_to_frontier, self.frontier)
        self._save(self.path_to_visited, self.visited_urls_in_queue)
        logging.info(f"[COMPLETE] {crawled} Seiten neu verarbeitet.")


if __name__ == '__main__':
    # 1) Define your seed URLs here:
    seeds = [
        "https://visit-tubingen.co.uk/welcome-to-tubingen/",
        "https://www.tuebingen.de/",
        "https://uni-tuebingen.de/en/",
        "https://en.wikipedia.org/wiki/T%C3%BCbingen",
        "https://www.germany.travel/en/cities-culture/tuebingen.html",
        "https://www.tripadvisor.com/Attractions-g198539-Activities-Tubingen_Baden_Wurttemberg.html",
        "https://www.europeanbestdestinations.com/destinations/tubingen/",
        "https://www.tuebingen.de/en/",
        "https://www.stadtmuseum-tuebingen.de/english/",
        "https://www.tuebingen-info.de/",
        "https://tuebingenresearchcampus.com/en/",
        "https://www.welcome.uni-tuebingen.de/",
        "https://integreat.app/tuebingen/en/news/tu-news",
        "https://tunewsinternational.com/category/news-in-english/",
        "https://historicgermany.travel/historic-germany/tubingen/",
        "https://visit-tubingen.co.uk/",
        "https://www.germansights.com/tubingen/",
        "https://www.tripadvisor.com/Restaurants-g198539-Tubingen_Baden_Wurttemberg.html",
        "https://www.tripadvisor.com/Restaurants-g198539-zfp58-Tubingen_Baden_Wurttemberg.html",
        "https://www.tripadvisor.com/Attractions-g198539-Activities-c36-Tubingen_Baden_Wurttemberg.html",
        "https://theculturetrip.com/europe/germany/articles/the-best-things-to-see-and-do-in-tubingen-germany",
        "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
        "https://justinpluslauren.com/things-to-do-in-tubingen-germany/",
        "https://simplifylivelove.com/tubingen-germany/",
        "https://wanderlog.com/list/geoCategory/199488/where-to-eat-best-restaurants-in-tubingen",
        "https://wanderlog.com/list/geoCategory/312176/best-spots-for-lunch-in-tubingen",
        "https://guide.michelin.com/us/en/baden-wurttemberg/tbingen/restaurants",
        "https://www.outdooractive.com/en/eat-and-drink/tuebingen/eat-and-drink-in-tuebingen/21873363/",
        "https://www.opentable.com/food-near-me/stadt-tubingen-germany",
        "https://www.1821tuebingen.de/",
        "https://www.historicgermany.travel/tuebingen/",
        "https://www.expedia.com/Things-To-Do-In-Tubingen.d55289.Travel-Guide-Activities",
        "https://www.lonelyplanet.com/germany/baden-wurttemberg/tubingen",
        "https://www.tripadvisor.com/Attraction_Review-g198539-d14983273-Reviews-Tubingen_Weinwanderweg-Tubingen_Baden_Wurttemberg.html",
        "https://www.yelp.com/search?find_desc=Restaurants&find_loc=T%C3%BCbingen",
        "https://guide.michelin.com/en/baden-wurttemberg/tubingen/restaurants",
        "https://wanderlog.com/geoInMonth/10053/7/tubingen-in-july",
        "https://www.wanderlog.com/list/geoCategory/76026/best-restaurants-in-tubingen",
        "https://en.wikivoyage.org/wiki/T%C3%BCbingen",
        "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Eat",
        "https://en.wikivoyage.org/wiki/T%C3%BCbingen#Drink",
        "https://www.mygermanyvacation.com/things-to-do-in-tubingen/"
    ]

    # 2) Instantiate and configure the crawler:
    #    max_depth: how many link‐hops you’ll follow
    #    delay: default crawl‐delay between requests (sec)
    #    simhash_threshold: minimal Hamming‐distanz to consider a page “changed”
    crawler = OfflineCrawler(
        seeds=seeds,
        max_depth=2,
        delay=0.5,
        simhash_threshold=3
    )

    # 3) Run! This will load any existing pickles, resume the frontier,
    #    respect Last-Modified headers + SimHash checks, and write back all pickles.
    crawler.run()
