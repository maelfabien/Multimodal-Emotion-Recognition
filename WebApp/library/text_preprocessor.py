from gensim.models import KeyedVectors
from gensim.models import word2vec

import numpy as np
import pandas as pd
import re
import datetime
from operator import itemgetter
from random import randint
import seaborn as sns

import os
import time
import string
import dill
import pickle

from nltk import *

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
from nltk.stem.snowball import SnowballStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split as tts
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, SpatialDropout1D, Activation, Conv1D, MaxPooling1D, Input, concatenate
from keras.utils.np_utils import to_categorical
from keras import backend as K

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
        Transforms input data by using NLTK tokenization, POS tagging, lemmatization and vectorization.
        """
    
    def __init__(self, max_sentence_len = 300, stopwords=None, punct=None, lower=True, strip=True):
        """
            Instantiates the preprocessor.
            """
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.max_sentence_len = max_sentence_len
    
    def fit(self, X, y=None):
        """
            Fit simply returns self.
            """
        return self
    
    def inverse_transform(self, X):
        """
            No inverse transformation.
            """
        return X
    
    def transform(self, X):
        """
            Actually runs the preprocessing on each document.
            """
        output = [(self.tokenize(doc)) for doc in X]
        return output
    
    def tokenize(self, document):
        """
            Returns a normalized, lemmatized list of tokens from a document by
            applying segmentation, tokenization, and part of speech tagging.
            Uses the part of speech tags to look up the lemma in WordNet, and returns the lowercase
            version of all the words, removing stopwords and punctuation.
            """
        lemmatized_tokens = []
        
        # Clean the text
        document = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", document)
        document = re.sub(r"what's", "what is ", document)
        document = re.sub(r"\'s", " ", document)
        document = re.sub(r"\'ve", " have ", document)
        document = re.sub(r"can't", "cannot ", document)
        document = re.sub(r"n't", " not ", document)
        document = re.sub(r"i'm", "i am ", document)
        document = re.sub(r"\'re", " are ", document)
        document = re.sub(r"\'d", " would ", document)
        document = re.sub(r"\'ll", " will ", document)
        document = re.sub(r"(\d+)(k)", r"\g<1>000", document)
        
        # Break the document into sentences
        for sent in sent_tokenize(document):
            
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                
                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue
                
                # Lemmatize the token
                lemma = self.lemmatize(token, tag)
                lemmatized_tokens.append(lemma)
        
        doc = ' '.join(lemmatized_tokens)
        #tokenized_document = self.vectorize(np.array(doc)[np.newaxis])
        return doc


    def vectorize(self, doc):
        """
            Returns a vectorized padded version of sequences.
            """
        save_path = "Models/padding.pickle"
        with open(save_path, 'rb') as f:
            tokenizer = pickle.load(f)

        doc_pad = tokenizer.texts_to_sequences(doc)
        return np.squeeze(doc_pad)

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform WordNet lemmatization.
        """
        tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)
