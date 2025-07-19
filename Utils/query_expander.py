from nltk.corpus import wordnet

class QueryExpander:
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
        return synonyms
