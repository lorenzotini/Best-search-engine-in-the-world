
import requests
from collections import deque
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import string
# from crawling import OfflineCrawler



# From the top 10 documents returned retrieve a dictionary of
# URL, title, publishing date, summary?

# crawl = OfflineCrawler()

def extract_publish_date(soup):
    # Try common meta tag patterns for publish date
    meta_props = [
        {'property': 'article:published_time'},
        {'name': 'pubdate'},
        {'name': 'publish-date'},
        {'name': 'timestamp'},
        {'name': 'date'},
        {'itemprop': 'datePublished'}
    ]
    for attrs in meta_props:
        tag = soup.find('meta', attrs=attrs)
        if tag and tag.get('content'):
            return tag['content']
    # Try <time> tag
    time_tag = soup.find('time')
    if time_tag and time_tag.get('datetime'):
        return time_tag['datetime']
    return None

def extract_summary(soup):
    # Check description meta tag
    desc = soup.find('meta', attrs={'name': 'description'})
    if desc and desc.get('content'):
        return desc['content']
    # Try Open Graph description
    og_desc = soup.find('meta', attrs={'property': 'og:description'})
    if og_desc and og_desc.get('content'):
        return og_desc['content']
    # Fallback to first paragraph
    # p = soup.find('p')
    # return p.get_text(strip=True) if p else None
    return None
    

def recrawl():

    docs = [("https://en.wikipedia.org/wiki/Okapi_BM25",0),
            ("https://medium.com/etoai/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6",1),
            ("https://weaviate.io/blog/hybrid-search-explained",2)]
    
    doc_info = dict()
    frontier = deque([x for x in docs])
    
    visited = set()

    while frontier:
        (url, doc_id) = frontier.popleft()
        url = url.strip()
        url = str(url)

        if url in visited:
            continue
        visited.add(url)


        try:
            resp = requests.get(url, headers={"Accept-Language": "en-US,en;q=0.9"}, timeout=5)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else None
        publish_date = extract_publish_date(soup)
        summary = extract_summary(soup)

        doc_info[doc_id] = {
            "url": url,
            "title": title,
            "publish_date": publish_date,
            "summary": summary
        }

    return doc_info


print(recrawl())