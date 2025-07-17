import requests
from bs4 import BeautifulSoup
from newspaper import Article

def extract_metadata(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    data = {
        "url": url,
        "title": None,
        "summary": None,
        "date": None
    }

    # Title from <title> or <meta property="og:title">
    title = soup.title.string if soup.title else None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]
    data["title"] = title

    # Summary from <meta name="description"> or <meta property="og:description">
    description = soup.find("meta", attrs={"name": "description"}) or \
                  soup.find("meta", property="og:description")
    if description and description.get("content"):
        data["summary"] = description["content"]

    # Date from common tags
    date = (
        soup.find("meta", {"name": "pubdate"}) or
        soup.find("meta", {"name": "publish-date"}) or
        soup.find("meta", {"property": "article:published_time"}) or
        soup.find("time")
    )
    if date:
        content = date.get("content") or date.get_text(strip=True)
        data["date"] = content
    else:
        a = Article(url)
        a.download()
        a.parse()
        data["date"] = a.publish_date

    return data
