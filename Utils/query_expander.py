from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

""" class QueryExpander:
    def __init__(self, max_synonyms=2):
        self.max_synonyms = max_synonyms

    def expand(self, tokens):
        expanded_tokens = set(tokens)

        for token in tokens:
            synonyms = self.get_synonyms(token)
            expanded_tokens.update(synonyms)

        return list(expanded_tokens)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_word = lemma.name().replace("_", " ")
                if syn_word != word:
                    synonyms.add(syn_word)
                if len(synonyms) >= self.max_synonyms:
                    break
            if len(synonyms) >= self.max_synonyms:
                break
        return synonyms """
    

stop_words = set(stopwords.words("english"))

class QueryExpander:
    def __init__(self, max_synonyms=2):
        self.max_synonyms = max_synonyms
        self.lemmatizer = WordNetLemmatizer()

    def expand(self, tokens):
        expanded_tokens = set(tokens)

        for token in tokens:
            synonyms = self.get_synonyms(token)
            expanded_tokens.update(synonyms)

        # Remove stopwords and lemmatize expanded terms
        final_tokens = [
            self.lemmatizer.lemmatize(t) for t in expanded_tokens 
            if t not in stop_words and len(t) > 2
        ]

        return final_tokens

    
    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_word = lemma.name().replace("_", " ").lower()
                if syn_word != word and " " not in syn_word:  # <--- Only one-word synonyms
                    synonyms.add(syn_word)
                if len(synonyms) >= self.max_synonyms:
                    break
            if len(synonyms) >= self.max_synonyms:
                break
        return synonyms

