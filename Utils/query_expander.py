import nltk
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))

class QueryExpander:
    def __init__(self, max_synonyms=2):
        self.max_synonyms = max_synonyms
        self.lemmatizer = WordNetLemmatizer()

    def expand(self, tokens):
        expanded_tokens = set(tokens)

        pos_tags = pos_tag(tokens)

        for token, pos in pos_tags:
            wn_pos = self.get_wordnet_pos(pos)
            if wn_pos in [wordnet.NOUN, wordnet.VERB]:
                synonyms = self.get_synonyms(token, wn_pos)
                expanded_tokens.update(synonyms)

        # POS-aware lemmatization and stopword filtering
        final_tokens = set()
        for token, pos in pos_tag(expanded_tokens):
            if token not in stop_words and len(token) > 2:
                wn_pos = self.get_wordnet_pos(pos)
                lemma = self.lemmatizer.lemmatize(token, wn_pos)
                final_tokens.add(lemma)

        return list(final_tokens)


    def get_synonyms(self, word, wn_pos):
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                syn_word = lemma.name().replace("_", " ").lower()
                if syn_word != word and " " not in syn_word:  # Keep only one-word terms
                    synonyms.add(syn_word)
                if len(synonyms) >= self.max_synonyms:
                    break
            if len(synonyms) >= self.max_synonyms:
                break
        return synonyms

    def get_wordnet_pos(self, tag):
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
