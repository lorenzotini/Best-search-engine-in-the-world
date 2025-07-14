import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Ensure stopwords are available
import nltk
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# TODO this function is taken mostly from crawling preprocessing function.
# i dont know if we should keep stopwords (think about negations in a query)
# or keep other information
def preprocess_query(text):
    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters and normalize whitespace
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Filter stopwords and lemmatize words
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return filtered_tokens