from .. import utils
from . import basic
from . import synonym_malefemale_replacement as syn
from . import lemmatizer
from . import stopwords

from nltk.stem.snowball import FrenchStemmer
import pandas as pd
import collections

# ---------------------------------------------------------------------------------------------------------------------------
# Pipeline Preprocessing
# ---------------------------------------------------------------------------------------------------------------------------


# Dictionnaire des fonctions de pré-processing disponibles
usage = {
    'notnull': basic.notnull,
    'remove_accents': basic.remove_accents,
    'remove_stopwords': basic.remove_stopwords,
    'trim_string': basic.trim_string,
    'remove_punct': basic.remove_punct,
    'pe_matching': basic.pe_matching,
    'to_lower': basic.to_lower,
    'to_lower_all': basic.to_lower_all,
    'remove_numeric': basic.remove_numeric,
    'remove_gender_synonyms': basic.remove_gender_synonyms,
    'lemmatize': basic.lemmatize,
    'lemmatize_ws': basic.lemmatize_ws,
    'stemmatize': basic.stemmatize,
    'add_point': basic.add_point,
    'add_space_around_special': basic.deal_with_specific_characters
}


default_pipeline = ['notnull', 'to_lower', 'pe_matching', 'trim_string',
                                        'remove_gender_synonyms', 'remove_punct', 'remove_numeric',
                                        'remove_stopwords','stemmatize', 'remove_accents']


# Chaîne de traitement de pré-processing
#
# A partir d'une pandas Series de documents, retourne une nouvelle
# pandas Series contenant les documents modifiés par les fonctions
# contenues dans pipeline
#
# @param docs           [pandas.Series] corpus en entrée
# @param pipeline       [str list] fonctions à appliquer (ayant une correspondance dans usage)
# @return corpus pré-processé
@utils.data_agnostic_function(modify_data=True, chunk_size=1000)
@utils.process_docs_keep_everything()
def preprocess_pipeline(docs, pipeline=default_pipeline):
    for item in pipeline:
        if item in usage.keys():
            docs = usage[item](docs)
    return docs


# Class PreProcessor compatible pipeline sklearn
#
# Cette classe implémente les méthodes fit & transform et permet
# d'appeler un pipeline de préprocessing dans un pipeline sklearn
#
# @method fit           Ne fait rien mais est requise pour être compatible sklearn
# @method transform     Wrapper pour preprocess_pipeline
class PreProcessor:
    def __init__(self, pipeline=default_pipeline):
        self.pipeline = pipeline

    def fit(self):
        pass

    def transform(self, docs):
        if type(docs) is str or type(docs) is list:
            docs = pd.Series(docs)
        return preprocess_pipeline(docs, pipeline=self.pipeline)


# Retourne une instance de PreProcessor
#
# @param pipeline     array[str]  Le pipeline de preprocessing à appliquer lors de l'appel de PreProcessor.transform
# @return             PreProcessor La classe wrapper
def get_preprocessor(pipeline=default_pipeline):
    return PreProcessor(pipeline)

# ---------------------------------------------------------------------------------------------------------------------------
# Fonctions listing_count_words, fonctions list_one_appearance_word, remove_words
# ---------------------------------------------------------------------------------------------------------------------------


## Liste de mots uniques
#
# Retourne un data frame donnant l'ensemble des mots apparaissant dans un corpus de texte ainsi que leurs fréquences
# Comptabilisation des occurences des mots + tri par fréquence décroissante (data frame en sortie)
#
# @param docs
# @return count_words Data Frame
def listing_count_words(docs):
    words = list([word for sentence in docs.str.split() for word in sentence])
    count_words = collections.Counter(words)
    count_words = pd.DataFrame(sorted(count_words.items(), key=lambda t: t[1], reverse=True),
                               columns=["word", "count"])

    return count_words


## Liste des mots d'occurence unique
#
# Retourne un listing des mots n'apparaissant qu'une seule fois dans un corpus de textes
#
# @param count_words    Dataframe de comptage des mots fourni par la fonction listing_count_words
# @return Liste des mots d'occurence unique
def list_one_appearance_word(count_words):
    stop_less_commun_words = count_words["word"][count_words["count"] <= 1]
    return stop_less_commun_words


# Filtre des mots
#
# Retourne les mots du texte non présents dans la liste de mots à enlever
#
# @param  text              [str] texte à filtrer
# @param words_to_remove    Liste de mots à enlever
# @return Liste des mots du texte filtrés
def remove_words(text, words_to_remove):
    return [w for w in text if w not in words_to_remove]


