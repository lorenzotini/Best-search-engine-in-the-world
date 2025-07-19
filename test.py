import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Feuerzangenbowle"

response = requests.get(url, timeout=0.2)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

def extract_title_from_soup(soup):
    """
    Extracts a meaningful title from a BeautifulSoup object.
    Tries multiple methods to find the best title.
    """

    # 1. <title> tag (basic HTML)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # 2. Open Graph metadata: <meta property="og:title" content="...">
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    # 3. Twitter Cards metadata: <meta name="twitter:title" content="...">
    twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
    if twitter_title and twitter_title.get("content"):
        title = twitter_title["content"].strip()

    return title



def extract_description_from_soup(soup):
    """
    Extracts a meaningful description from a BeautifulSoup object.
    Tries meta tags first, then finds the first non-empty <p> tag.
    """

    # Step 1: Try meta tags
    meta_keys = [
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ]

    for attrs in meta_keys:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"].strip()

    # Step 2: Look in the main content area
    content = soup.find("div", id="mw-content-text") or soup.body
    if content:
        for p in content.find_all("p"):
            text = p.get_text(strip=True)
            # Skip if paragraph is empty or just punctuation/refs
            if text and len(text) > 40:  # Adjust length threshold as needed
                return text

    return "No description available."


print("Title:", title, "\nDescription:", extract_description_from_soup(soup))