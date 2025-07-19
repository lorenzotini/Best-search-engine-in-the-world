# Flask version of your Streamlit app

from flask import Flask, render_template, request, send_file
import requests
from bs4 import BeautifulSoup
from main import search
import numpy as np
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# ---------------- Helper Functions ----------------

def document_sentiment_analysis_binary(data : list[str], pipeline, seed= 0, random_aprox=False):

    doc_analysis = {}

    if random_aprox:

        np.random.seed(seed)
        # natural random numbers for testing
        random_scores = np.unique(np.random.randint(0, len(data), 10))
        data = [data[i] for i in random_scores]

    analysis = pipeline(data)

    if analysis["label" == "NEGATIVE"] != None:
        doc_analysis["negative"] = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "NEGATIVE" ]) / len(analysis)
    else:
        doc_analysis["negative"] = 0
        
    if analysis["label" == "NEUTRAL"] != None:
        doc_analysis["neutral"] = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "neutral" ]) / len(analysis)
    else:
        doc_analysis["negative"] = 0
    if analysis["label" == "POSITIVE"] != None:
        doc_analysis["positive"] = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "POSITIVE" ])  / len(analysis)
    else:
        doc_analysis["positive"] = 0

    max_key = max(doc_analysis.items(), key=lambda item: item[1])[0]
    max_value = doc_analysis[max_key]

    return {"label": max_key, "score": max_value}

def preprocess_text(text: str, chunk_size: int = 250):
    # # Ensure stopwords are available
    # stop_words = set(stopwords.words("english"))

    # Lowercase and clean text
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    document = [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]


    if max([len(doc.split()) for doc in document]) > chunk_size:
        ValueError("Document splitting failed, some chunks are larger than the specified chunk size.")

    return document

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

def get_document_data(url, pipeline):
    try:
        response = requests.get(url, timeout=0.2)
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    title = extract_title_from_soup(soup)
    description = extract_description_from_soup(soup)

    text = preprocess_text(soup.get_text(separator=" ", strip=True))
    sentiment_analy = document_sentiment_analysis_binary(text, pipeline, seed=0, random_aprox=True)

    return {
        "title": str(title),
        "url": url,
        "description": str(description) if description else "",
        "sentiment": sentiment_analy["label"],
        "sentiment_score": int(sentiment_analy["score"] * 100),
    }


def get_results(query, sentiment_filter=None):
    results_urls = search(query)

    # get sentiment analysis pipeline   
    sentiment_pipeline = pipeline("sentiment-analysis")

    data = [get_document_data(url, sentiment_pipeline) for url in results_urls[:10]]  # Limit to first 2 URLs for demo

    # Remove None results (failed requests)
    data = [result for result in data if result is not None]

    # sentiment filter
    if sentiment_filter:
        data = [result for result in data if result['sentiment'] == sentiment_filter]

    return data

# ---------------- Routes ----------------

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    results = []
    sentiment_filter = ""
    if request.method == 'POST':
        query = request.form.get('query')
        sentiment_filter = request.form.get('sentiment_filter')
        if query:
            if sentiment_filter:
                mock_data = get_results(query, sentiment_filter)
            else:
                mock_data = get_results(query)
            results = mock_data

    return render_template('index.html', query=query, results=results, sentiment_filter=sentiment_filter)




# ---------------- Run App ----------------

if __name__ == '__main__':
    app.run(debug=True)
