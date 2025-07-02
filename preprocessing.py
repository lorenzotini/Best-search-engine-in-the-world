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
    token_to_ids = {}

    # Process every document
    for doc in docs:

        # Extract raw text from document and preprocess it into tokens
        tokens = preprocess_text(doc[0])

        # Save document ID
        doc_id = doc[1]

        # For every token, create a list containing the IDs of the documents in which the token is present
        for token in tokens:
            if token_to_ids.get(token) is None:    # First time we see this term -> initialize it
                token_to_ids[token] = [doc_id]
            elif doc_id in token_to_ids[token]:
                continue
            else:
                token_to_ids[token].append(doc_id)
    
    # Add skip pointers
    for token in token_to_ids:
        current_ids = token_to_ids[token]
        
        # Insert a pointer every pointer_freq entry
        pointer_freq = math.ceil(math.sqrt(len(token_to_ids[token])))

        skip_pointers = []
        for i, id in enumerate(current_ids):
            index_to_skip = None
            doc_at_that_index = None
            if i % pointer_freq == 0:
                index_to_skip = i + pointer_freq
                if index_to_skip >= len(current_ids):
                    index_to_skip = len(current_ids) - 1
                doc_at_that_index = current_ids[index_to_skip]
            skip_pointers.append([id, index_to_skip, doc_at_that_index])
        
        skip_dict[token] = skip_pointers



skip_dict = {}
corpus = [
    ['The government released new policy data on cybersecurity and public infrastructure today in a national press briefing.', 0],
    ['Experts argue the policy reform is essential for national security and government system resilience against threats.', 1],
    ['New data shows that cybersecurity investments improved government system performance and reduced downtime significantly.', 2],
    ['The government system experienced a data breach despite existing policy and updated firewall security implementations.', 3],
    ['Security officials demand stronger policy and encryption standards to protect national data and government records.', 4],
    ['Today’s briefing emphasized the role of AI in enhancing security, policy enforcement, and data processing within the government.', 5],
    ['Public trust in the government system depends on transparent policy updates and strong digital data protection.', 6],
    ['The minister highlighted government achievements in digital infrastructure and national policy improvements on security.', 7],
    ['Despite increased investment in cybersecurity, data leaks remain a threat to national policy and government transparency.', 8],
    ['Government agencies collaborated on a new system to monitor policy compliance and analyze real-time data for threats.', 9]
]


save_document_skip_pointers(corpus, skip_dict)

for entry, value in skip_dict.items():
    print(entry, value)