import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import math

""" nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
 """
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML using BeautifulSoup (more robust)
    text = BeautifulSoup(text, "html.parser").get_text()

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters and normalize whitespace
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Filter stopwords and stem words
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return filtered_tokens


def save_document_skip_pointers(docs, skip_dict):
    for doc in docs:
        tokens = preprocess_text(doc[0])
        doc_id = doc[1]
        for token in tokens:
            if skip_dict.get(token) is None:    # First time we see this term -> initialize it
                skip_dict[token] = [doc_id]
            elif doc_id in skip_dict[token]:
                continue
            else:
                skip_dict[token].append(doc_id)
    
"""     for token in skip_dict:
        pointer_freq = math.ceil(math.sqrt(len(skip_dict[token]))) """
        



skip_dict = {}
corpus = [['The government is open.', 0], 
          ['The government is closed.', 1], 
          ['Long live Mickey Mouse, emperor of all!', 2], 
          ['Darn! This will break.', 3]]

save_document_skip_pointers(corpus, skip_dict)

for entry, value in skip_dict.items():
    print(entry, value)