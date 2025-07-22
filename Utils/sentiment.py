import numpy as np
from transformers import pipeline

sentiment_pipeline_positive_negative = pipeline("sentiment-analysis")

pipe_subjectivity = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english")


# load data from a file as junks inta array

def join_data(tokens, chunk_size=512):
    return [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]


def document_sentiment_analysis_binary(data : list[str], pipeline=sentiment_pipeline, seed= 0, random_aprox=False):

    doc_analysis = {}

    if random_aprox:

        np.random.seed(seed)
        # natural random numbers for testing
        random_scores = np.unique(np.random.randint(0, len(data), 10))
        data = [test_data[i] for i in random_scores]

    analysis = pipeline(data)

    if analysis["label" == "NEGATIVE"] != None:
        negative_prob = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "NEGATIVE" ]) / len(analysis)
    else:
        negative_prob = 0
    if analysis["label" == "POSITIVE"] != None:
        positive_prob = np.sum([doc_analysis["score"] for doc_analysis in analysis if doc_analysis["label"] == "POSITIVE" ])  / len(analysis)
    else:
        positive_prob = 0

    if negative_prob > positive_prob:
        doc_analysis["label"] = "NEGATIVE"
        doc_analysis["score"] = negative_prob
    else:
        doc_analysis["label"] = "POSITIVE"
        doc_analysis["score"] = positive_prob

    return doc_analysis