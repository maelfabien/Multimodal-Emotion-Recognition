from Text.Python.load_data import *
from Text.Python.train import *

from nltk.corpus import movie_reviews as reviews
from sklearn.datasets import fetch_20newsgroups
from gensim.models import KeyedVectors
from gensim.models import word2vec

import numpy as np
import pandas as pd
import re
import datetime
from operator import itemgetter
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
import random

import os
import time
import string
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
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, SpatialDropout1D, Activation, Conv1D, MaxPooling1D, Input, concatenate
from keras.utils.np_utils import to_categorical

class visualize:

    def __init__(self, complete_dataset, X, labels_list):
        self.data = complete_dataset
        self.X = X
        self.labels_list = labels_list

    def textlength_vs_labels_histogram(self):
        # Visualization of histograms of text length vs. label
        for label in self.labels_list:
            g = sns.FacetGrid(data=self.data, col=label)
            g.map(plt.hist, 'text length', bins=50)
        plt.show()

    def textlength_vs_labels_boxplot(self):
        # Visualization of boxplots of text length vs. label
        for i, label in enumerate(self.labels_list):
            plt.figure(i)
            sns.boxplot(x=label, y='text length', data=self.data)
        plt.show()

    def most_frequent_words(self):
        # Visualization of the most frequent words
        complete_corpus = ' '.join(self.X)
        words = tokenize.word_tokenize(complete_corpus)
        fdist = FreqDist(words)
        print("List of 100 most frequent words/counts")
        print(fdist.most_common(100))
        fdist.plot(40)

    def most_frequent_words_preprocessed(self):
        # Visualization of the most frequent words
        if not hasattr(self, 'X_preprocess'):
            preprocessor = train(corpus = self.X).NLTKPreprocessor
            self.X_preprocess = prep.transform(self.X).tolist()
        complete_corpus = ' '.join(self.X_preprocess)
        words = tokenize.word_tokenize(complete_corpus)
        fdist = FreqDist(words)
        print("List of 100 most frequent words/counts")
        print(fdist.most_common(100))
        fdist.plot(40)

    def get_corpus_statistics(self):
        # Retrieve some info on the text data
        numWords = []
        for text in self.X:
                counter = len(text.split())
                numWords.append(counter)  
        numFiles = len(numWords)
        print('The total number of essays is', numFiles)
        print('The total number of words in all essays is', sum(numWords))
        print('The average number of words in each essay is', sum(numWords)/len(numWords))

    def get_preprocessed_corpus_statistics(self):
        # Retrieve some info on the preprocessed text data
        if not hasattr(self, 'X_preprocess'):
            preprocessor = train(corpus = self.X).NLTKPreprocessor
            self.X_preprocess = prep.transform(self.X).tolist()
        len_list = [np.count_nonzero(self.X_preprocess[i]) for i in range(len(self.X))]
        print('The average number of words in each preprocessed essay is', np.mean(len_list))
        print('The standard deviation of the number of words in each preprocessed essay is', np.std(len_list))
        print('The average number of words in each preprocessed essay plus 2 standard deviations is', np.mean(len_list) + 2 * np.std(len_list))

class tsne:
    
    def __init__(self, X, max_features = 30000, max_sentence_len = 300, embed_dim = 300,  n_elements = 100):
        self.X = X
        self.max_features =max_features
        self.max_sentence_len = max_sentence_len
        self.embed_dim = embed_dim
        self.n_elements = n_elements
        self.vectors, self.words, self.dic =  self.prepare_embedding(self.X)

    def load_google_vec(self):
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        #wget.download(url, 'Data/GoogleNews-vectors.bin.gz')
        return KeyedVectors.load_word2vec_format(
            'Data/GoogleNews-vectors.bin.gz',
            binary=True)

    def lemmatize_token(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        return WordNetLemmatizer().lemmatize(token, tag)


    def get_preprocessed_corpus(self, X_corpus):
        """
        Returns a preprocessed version of a full corpus (ie. tokenization and lemmatization using POS taggs)
        """
        X = ' '.join(X_corpus)
        lemmatized_tokens = []

        # Break the document into sentences
        for sent in sent_tokenize(X):

            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):

                # Apply preprocessing to the token
                token = token.lower()
                token = token.strip()
                token = token.strip('_')
                token = token.strip('*')

                # If punctuation or stopword, ignore token and continue
                if token in set(sw.words('english')) or all(char in set(string.punctuation) for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize_token(token, tag)
                lemmatized_tokens.append(lemma)

        doc = ' '.join(lemmatized_tokens)
        return doc


    def prepare_embedding(self, X):
        """
        Returns the embedding weights matrix, the word index, and the word-vector dictionnary corresponding
        to the training corpus set of words.
        """
        # Load Word2Vec vectors
        word2vec = self.load_google_vec()

        # Fit and apply an NLTK tokenizer on the preprocessed training corpus to obtain sequences.
        tokenizer = Tokenizer(num_words=self.max_features)
        X_pad = self.get_preprocessed_corpus(X)
        tokenizer.fit_on_texts(pd.Series(X_pad))
        X_pad = tokenizer.texts_to_sequences(pd.Series(X_pad))

        # Pad the sequences
        X_pad = pad_sequences(X_pad, maxlen=self.max_sentence_len, padding='post', truncating='post')

        # Retrieve the word index
        train_word_index = tokenizer.word_index

        # Construct the embedding weights matrix and word-vector dictionnary
        train_embedding_weights = np.zeros((len(train_word_index) + 1, self.embed_dim))
        for word, index in train_word_index.items():
            train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(self.embed_dim)
        word_vector_dict = dict(zip(pd.Series(list(train_word_index.keys())),
                                    pd.Series(list(train_word_index.keys())).apply(
                                        lambda x: train_embedding_weights[train_word_index[x]])))
        return train_embedding_weights, train_word_index, word_vector_dict


    def plot(self):
        labels = []
        tokens = []

        l_bound = 0
        u_bound = len(self.words)
        step = int(len(self.words)/self.n_elements)

        #for index in range(l_bound,u_bound, step):
        for index in random.sample(range(l_bound,u_bound), self.n_elements):
            tokens.append(self.vectors[index])
            labels.append(self.words[index])

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        xx = []
        yy = []
        for value in new_values:
            xx.append(value[0])
            yy.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(xx)):
            plt.scatter(xx[i],yy[i])
            plt.annotate(labels[i],
                         xy=(xx[i], yy[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, POS tagging, lemmatization and vectorization.
    """

    def __init__(self, corpus, max_sentence_len = 300, stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor.
        """
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.corpus = corpus
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
        output = np.array([(self.tokenize(doc)) for doc in X])
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
        tokenized_document = self.vectorize(np.array(doc)[np.newaxis])
        return tokenized_document


    def vectorize(self, doc):
        """
        Returns a vectorized padded version of sequences.
        """
        save_path = "Data/padding.pickle"
        with open(save_path, 'rb') as f:
            tokenizer = pickle.load(f)
        doc_pad = tokenizer.texts_to_sequences(doc)
        doc_pad = pad_sequences(doc_pad, padding='pre', truncating='pre', maxlen=self.max_sentence_len)
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
