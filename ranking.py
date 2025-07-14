import string
import math


Q = 'french bulldog'
# D = ['the french revolution was a period of upheaval in france', 
    #  'the french bulldog is a small breed of domestic dog', 
    #  'french is a very french language spoken by the french']

D = [ 'She couldnt understand why nobody else could see that the sky is full of cotton candy.',
        'The tears of a clown make my lipstick run, but my shower cap is still intact.',
        'Acres of almond trees lined the interstate highway which complimented the crazy driving nuts.',
        'They finished building the road they knew no one would ever use.', 
        'For oil spots on the floor, nothing beats parking a motorbike in the lounge.',
        'For oil spots on the floor, nothing beats parking a motorbike in the lounge.',
        'He had unknowingly taken up sleepwalking as a nighttime hobby.',
        'If eating three-egg omelets causes weight-gain, budgie eggs are a good substitute.',
        'He walked into the basement with the horror movie from the night before playing in his head.',
        'Going from child, to childish, to childlike is only a matter of time.'


     ]


# TF function
def TF1(Q, D):
    sum = 0
    tokens = Q.split()
    words = D.split()
    for t in tokens:
        for w in words:
            if t==w:
                sum += 1

    return sum

# TF(Q,D[2])


#Turn a corpus of arbitrary texts into term-frequency weighted BOW vectors.
def TF(corpus):
    vecs = []
    for c in corpus:
        # print(c)
        # prepare text removing punctuation and splitting
        text = c

        text = ''.join([char for char in text if char not in string.punctuation])
        splitted_text = text.split()

        # compute bow
        bow = {}
        for word in splitted_text:
            if bow.get(word) is None:
                bow[word] = 1
            else:
                bow[word] += 1
        vecs.append(bow)
    print(vecs)
    return vecs


# print(TF1(D))





#Estimate inverse document frequencies based on a corpus of documents.
def IDF(corpus):
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

# print(IDF(D))

#Turn a corpus of arbitrary texts into TF-IDF weighted BOW vectors.
def TFIDF(corpus):
    # vecs is a list of dictionaries, each dictionary corresponds to a document, in each dict there is the map(term_in_this_doc -> TF-IDF)
    vecs = []
    idfs = IDF(corpus)
    for c in corpus:
        text = ''.join([char for char in c if char not in string.punctuation])
        splitted_text = text.split()
        
        # Count word frequencies in this document
        term_freq = {}
        for word in splitted_text:
            if term_freq.get(word) is None:
                term_freq[word] = 1
            else:
                term_freq[word] += 1
        
        # Create the actual tfidf mapping for this document
        tfidf = {}
        for k, v in term_freq.items():
            tfidf[k] = v * idfs[k]
        
        vecs.append(tfidf)

    return vecs

# print(TFIDF(D))


















# TF-IDF function


def preprocess(text):
    return ''.join([char for char in text if char not in string.punctuation]).split()






# BM25 function
# Inputs : Query, Document 

# Parameters : b controls strength of document length normalization
#               k controls rate of term saturation




# Output: Score of doc relevance to the query 
def BM25(Q, doc, k=1.5, b=0.75):
    # parse query 
    terms = preprocess(Q)

    # preprocess the text so we have a fair comparison between terms

    total_score = 0
    # for each term in the query 
    for t in terms:   
        # score = IDF(t)  #  IDF score
        nom = TF(t, doc) * (k+1)
        denom = TF(t, doc) + k*(1-b + b())

        total_score += IDF(t) * (nom/denom)

    return total_score


BM25('french bulldog')


# TO DO 
# if possible do stuff beforehand
#   compute IDF(q) after indexing before calling ranking
#   preprocess the document
# search optimal parameters 