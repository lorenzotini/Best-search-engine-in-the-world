# Flask version of your Streamlit app

from flask import Flask, render_template, request, send_file
import requests
from bs4 import BeautifulSoup
from main import search, init_search
import numpy as np
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from concurrent.futures import ThreadPoolExecutor
import time

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

    if analysis["label" == "LABEL_0"] != None:
        doc_analysis["objective"] = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "LABEL_0" ]) / len(analysis)
    else:
        doc_analysis["objective"] = 0

    if analysis["label" == "LABEL_1"] != None:
        doc_analysis["subjective"] = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "LABEL_1" ])  / len(analysis)
    else:
        doc_analysis["subjective"] = 0

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
        response = requests.get(url, timeout=0.5)
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


def get_results(query, sentiment_pipeline, indexer, hybrid_model, sentiment_filter=None):
    start_time = time.time()
    results_urls = search(query, indexer, hybrid_model, use_query_expansion=True)

    data = [get_document_data(url, sentiment_pipeline) for (url, score) in results_urls[:10]]  # Limit to first 2 URLs for demo

    # Remove None results (failed requests)
    data = [result for result in data if result is not None]

    # sentiment filter
    if sentiment_filter:
        data = [result for result in data if result['sentiment'] == sentiment_filter]

    end_time = time.time()  # End timer
    search_duration = round(end_time - start_time,2)
    print(f"Search took {search_duration:.2f} seconds")  # Print or log the duration

    return data, search_duration

# ---------------- Routes ----------------

indexer = None
bm25_model = None
hybrid_model = None
sentiment_pipeline= None

with app.app_context():
    app.config["search_models"] = init_search()
    print("Search engine initialized with models.")


@app.route('/', methods=['GET', 'POST'])
def index():

    # Initialize models
    indexer, hybrid_model, sentiment_pipeline = app.config["search_models"]

    query = ""
    results = []
    sentiment_filter = ""
    search_duration = ""
    if request.method == 'POST':
        query = request.form.get('query')
        sentiment_filter = request.form.get('sentiment_filter')
        if query:
            if sentiment_filter:
                data, search_duration = get_results(query, sentiment_pipeline, indexer, hybrid_model, sentiment_filter)
            else:
                data, search_duration = get_results(query, sentiment_pipeline, indexer, hybrid_model)
            results = data

    return render_template('index.html', query=query, results=results, sentiment_filter=sentiment_filter, search_duration=search_duration)


# ---------------- Run App ----------------

if __name__ == '__main__':

    app.run(debug=False)