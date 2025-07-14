import string, math

def build_TF(corpus):
    tf_list = []
    for doc in corpus:
        text = doc[0]
        # remove punctuation and lowercase
        text = ''.join([char for char in text if char not in string.punctuation])
        splitted_text = text.lower().split()
        
        # build term-frequency dictionary
        bow = {}
        for word in splitted_text:
            bow[word] = bow.get(word, 0) + 1
        
        tf_list.append(bow)
    return tf_list


# Query function: term frequency in document d
def TF(t, d):
    return tf_data[d].get(t.lower(), 0)

#Estimate inverse document frequencies based on a corpus of documents.
def build_IDF(corpus):
    idfs = {}
    D = len(corpus)
    # Dts = {key: str, value: (Dt_value: int, seen_in_this_doc: bool)}
    Dts = {}
    for c in corpus:
        text = ''.join([char for char in c if char not in string.punctuation])
        splitted_text = text.split()

        # Reset visited values, since we are visiting a new document
        for key, (num, _) in Dts.items():
            Dts[key] = (num, False)

        for word in splitted_text:
            if Dts.get(word) is None:    # First time we see this term -> initialize it
                Dts[word] = (1, True)
            elif Dts[word][1] is False:  # Dt already contains an entry for this term but is the first time we see it in this doc
                Dts[word] = (Dts[word][0] + 1, True)   # Increment its frequency and toggle bool
            elif Dts[word][1] is True:
                continue                # We've already seen this term in this doc
            else:
                raise Exception('Something went wrong.')
    
    for k, v in Dts.items():
        Dt = v[0]
        idfs[k] = math.log(D/Dt, 10)
        
    return idfs


def compute_doc_lengths(tf_data):
    return [sum(doc.values()) for doc in tf_data]


def bm25_score(query, doc_index, tf_data, idfs, doc_lengths, avgdl, k1=1.5, b=0.75):
    score = 0.0
    doc_len = doc_lengths[doc_index]
    for term in query:
        tf = tf_data[doc_index].get(term.lower(), 0)
        idf = idfs.get(term, 0)
        denom = tf + k1 * (1 - b + b * doc_len / avgdl)
        if denom > 0:
            score += idf * ((tf * (k1 + 1)) / denom)
    return score


def bm25_ranking(query, corpus, tf_data, idfs):
    doc_lengths = compute_doc_lengths(tf_data)
    avgdl = sum(doc_lengths) / len(doc_lengths)

    scores = []
    for i in range(len(corpus)):
        score = bm25_score(query, i, tf_data, idfs, doc_lengths, avgdl)
        scores.append((i, score))
    
    # Sort by score descending
    return sorted(scores, key=lambda x: x[1], reverse=True)


corpus = [['The government is is open.'], 
          ['The government is closed.'], 
          ['Long live Mickey Mouse, emperor of all!'], 
          ['Darn! This will break.']]


tf_data = build_TF(corpus)

idfs = build_IDF(corpus)

query = ['government', 'open']

ranking = bm25_ranking(query, corpus, tf_data, idfs)

for doc_idx, score in ranking:
    print(f"Document {doc_idx}: BM25 score = {score:.4f}")