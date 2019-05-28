from gensim.models import KeyedVectors
from gensim.models import word2vec

import numpy as np
import pandas as pd
import re
import datetime
from operator import itemgetter
from random import randint
import seaborn as sns
import wget

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
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, SpatialDropout1D, Activation, Conv1D, MaxPooling1D, Input, concatenate
from keras.utils.np_utils import to_categorical

class train_svm:

    def __init__(self, corpus):
        self.max_sentence_len = 300
        self.max_features = 300
        self.embed_dim = 300
        self.NLTKPreprocessor = self.NLTKPreprocessor(corpus)
        #self.MyRNNTransformer = self.MyRNNTransformer()


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


    class MyRNNTransformer(BaseEstimator, TransformerMixin):
        """
        Transformer allowing our Keras model to be included in our pipeline
        """
        def __init__(self, classifier):
            self.classifier = classifier

        def fit(self, X, y):
            batch_size = 32
            num_epochs = 35
            batch_size = batch_size
            epochs = num_epochs
            self.classifier.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
            return self

        def transform(self, X):
            self.pred = self.classifier.predict_proba(X)
            self.classes = [[0 if el < 0.2 else 1 for el in item] for item in self.pred]
            return self.classes


    class TfidfEmbeddingVectorizer(object):
        def __init__(self, word2vec):
            self.word2vec = word2vec
            self.word2weight = None
            self.dim = len(word2vec.values())

        def fit(self, X, y):
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words. So the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

            return self

        def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


    def identity(self, arg):
        """
        Simple identity function works as a passthrough.
        """
        return arg


    def reshape_a_feature_column(self, series):
        return np.reshape(np.asarray(series), (len(series), 1))


    def pipelinize_feature(self, function, active=True):
        def list_comprehend_a_function(list_or_series, active=True):
            if active:
                processed = [function(i) for i in list_or_series]
                processed = self.reshape_a_feature_column(processed)
                return processed
            else:
                return self.reshape_a_feature_column(np.zeros(len(list_or_series)))


    def get_text_length(self, text):
        return len(text)

    def load_google_vec(self):
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        wget.download(url, 'Data/GoogleNews-vectors.bin.gz')
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



    def multiclass_accuracy(self,predictions, target):
        "Returns the multiclass accuracy of the classifier's predictions"
        score = []
        for j in range(0, 5):
            count = 0
            for i in range(len(predictions)):
                if predictions[i][j] == target[i][j]:
                    count += 1
            score.append(count / len(predictions))
        return score


    def run(self,X, y, classifier=SGDClassifier, model_name=None,
                           verbose=True):
        """
        Builds a classifer for the given list of documents and targets
        """
        def build(classifier, X, y, embedding_dict, corpus):
            """
            Inner build function that builds a single model.
            """
            classifier = OneVsRestClassifier(classifier(), n_jobs=-1)
            model = Pipeline([
                ('preprocessor', self.NLTKPreprocessor),
                ("wordVectz", self.TfidfEmbeddingVectorizer(embedding_dict)),
                ('clf', classifier)
            ])
            return model.fit(X, y)

        y_trans = y

        # Prepare the embedding
        train_embedding_weights, train_word_index, wv_dict = self.prepare_embedding(X)

        # Begin evaluation
        if verbose: print("Building complete model and saving ...")
        model = build(classifier, X, y_trans, wv_dict, corpus=X)
        
        # Save the model
        if model_name:
            outpath = 'Models/'
            with open(outpath + model_name, 'wb') as f:
                dill.dump(model, f)
            print("Model written out to {}".format(model_name))

        return model
