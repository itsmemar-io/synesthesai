import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import string

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def clean(text):
    ignore = ',.:;?!\''
    music_parts = ['Intro','Outro','Pre-Chorus 1','Verse 1','Verse 1','Verse 1','Verse 1','Verse','Chorous',]
    for char in string.punctuation:
        if char not in ignore:
            text = text.replace(char, ' ') # Remove Punctuation
    for word in music_parts:
        text.replace(word, '')
    text = re.sub(r'\r\n', '. ', text)
    text = re.sub(r'\n', '. ', text)
    text = text.replace('  ', ' ')
    return text

def read_text(text):
    filedata = text
    article = filedata.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n=1):
    stop_words = stopwords.words('english')
    summarize_text = []

    #Read text and split it
    sentences =  read_text(text)

    #Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    #Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    #Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)

def summarizeLyrics(lyrics):
    clean_text = clean(lyrics)
    summary = generate_summary(clean_text)
    res = [clean_text, summary]
    print(res[0])
    print(res[1])
    return res
