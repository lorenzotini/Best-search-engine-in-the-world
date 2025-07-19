import re
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import logging
from langdetect import detect_langs


# Ensure stopwords are available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


# TODO ensure pos tag is downloaded
try:
    pos_tag([""])
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Helper function to map NLTK POS to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun
    

# TODO this function is taken mostly from crawling preprocessing function.
# i dont know if we should keep stopwords (think about negations in a query)
# or keep other information
def preprocess_text(text: str, isQuery=False):
    
    if not isQuery:
        # We are processing a BeautifulSoup object
        # Extract visible text (from the original soup, not the one modified for SimHash)
        for tag in text(["script", "style"]):
            tag.decompose()
        text = text.get_text(separator=" ")

        # Language filter
        langs = []
        try:
            langs = detect_langs(text)
        except:
            pass
        if not any(l.lang == "en" and l.prob >= 0.9 for l in langs):
            logging.warning("non-English page, skipping.")
            return None # Return None to indicate this page should be skipped for indexing
    else:
        # It is a query, so add tuebingen token to make related content more relevant
        # TODO how should we write tubingen? with ü or ue?
        text += " Tuebingen"
    
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    if isQuery:
        print("\nQUERY TOKENS before lemmat: ", tokens, "\n")

    # POS tagging
    pos_tags = pos_tag(tokens)

    # Lemmatization with correct POS
    filtered_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
        if token not in stop_words and len(token) > 2
    ]

    if isQuery:
        print("\nQUERY TOKENS: ", filtered_tokens, "\n")

    return filtered_tokens
