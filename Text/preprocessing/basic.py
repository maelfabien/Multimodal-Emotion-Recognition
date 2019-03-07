from . import synonym_malefemale_replacement as syn
from . import lemmatizer
from . import stopwords
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
import re
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# FONCTIONS diverses de preprocessing
# ----------------------------------------------------------------------------------------------------------------------

## Enlève les éléments null des documents
#
# Remplace les éléments null de docs par des chaines vides
#
# @param docs   Document à modifier
# @return Document modifié avec les null remplacés par des chaines vides
def notnull(docs):
    docs_notnull = docs.fillna('')
    return docs_notnull


# Transforme les mots ayant au moins 2 alphabétiques consécutifs en minuscules
# Cette approche permet de garder certains symboles (R ou C par exemple) en masjuscules et s'assurer
# qu'ils ne soient pas remove comme stopwords
#
# @param docs   Document à modifier
# @return Document modifié, sans majuscules

def to_lower(docs):
    return docs.apply(lambda x: " ".join(x.lower() if len(x)>1 else x for x in x.split()))


# Transforme les mots en minuscules
#
# @param docs   Document à modifier
# @return Document modifié, sans majuscules
def to_lower_all(docs):
    return docs.apply(lambda x: " ".join(x.lower() for x in x.split()))


# Permet de remplacer certaines expréssions / mots en dur
# Par exemple 'permis b' => 'permisb'
#
# @param docs   Document à modifier
# @return Document modifié, avec les transformations appliquées

def pe_matching(docs):
    docs = docs.str.replace('permis b', 'permisb')
    return docs


# Retourne un texte sans ponctuation. Si del_parenthesis == False,
# on les laisse dans les données (utilisé avant synonym_malefemale_replacement
#
# @param docs   Document à modifier
# @param del_parenthesis Si on supprime également les parenthèses et "/\"
# @return Document modifié, sans ponctuation

def remove_punct(docs, del_parenthesis=False):
    docs = docs.str.replace('[\',;:.!\*-?]', ' ')  # On ne retire NI le / NI les parenthèses dans un premier temps
    if del_parenthesis:
        docs = docs.str.replace('[()\/]', ' ')
    return docs

## Enlève les espaces multiples
#
# Remplace les espaces multiples par un seul espace
#
# @param docs   Document à modifier
# @return Document modifié

def trim_string(docs):
    return docs.str.replace('\s{2,}', ' ')


# Retourne un texte sans caractères numériques
#
# @param docs   Document à modifier
# @return Document modifié, sans caractères numériques

def remove_numeric(docs):
    return docs.str.replace('([0-9]+)', ' ')


# Retourne le texte sans les stopwords
# CF pe_semantic.preprocessing.stopwords pour les différents ensembles disponibles
#
# @param docs   Document à modifier
# @return Document modifié, sans stopwords

def remove_stopwords(docs):
    return docs.apply(lambda x: " ".join(stopwords.remove_stopwords(x.split(), opt='all')))


# Retourne un texte sans accents
#
# @param docs   Document à modifier
# @return Document modifié, sans accents

def remove_accents(docs):
    return docs.apply(lambda x: stopwords.remove_accents(str(x)))


##  Supression des synonymes de genre
#
# Trouve les équivalents féminins / masculins des noms de métier
# Par convention, conserve le masculin ( accord avec lemmatisation)
# cette fonction se base sur la présence de ponctuation, ne pas supprimer cette dernière
#
# /!\ TOFIX     : Le dataframe doit avoir une indexation de base commençant à 0 jusqu'à len(docs)-1
# >>> QUICKFIX  : Si ce n'est pas le cas, faire docs = docs.reset_index(drop = True)
#
# @param docs               Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def remove_gender_synonyms(docs):
    d = syn.malefemale_listing()
    liste_syn = d.listing_synonym(docs)  # On commence par définir la liste des synonymes
    d = syn.replace_synonym()
    docs = d.doc_replace_synonym(docs, liste_syn)
    return docs


def lemmatize(docs):
    docs = pd.Series(docs).str.replace('[()\/]', ' ')
    docs = lemmatizer.lemmatizer(docs)
    return docs


def lemmatize_ws(docs):
    docs = pd.Series(docs).str.replace('[()\/]', ' ')
    docs = lemmatizer.lemmatizer_ws(docs)
    return docs


##  Stemmatisation des mots
#
# Stemmatise les mots d'un corpus de texte soit :
# Supprime les suffixes des mots en fonction de règles préfédinies
# Afin de retrouver la racine des mots
#
# @param docs               Le corpus à nettoyer
# @return Le corpus une fois ses textes nettoyés
def stemmatize (docs):
    stemmer = FrenchStemmer()
    try:
        return docs.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))
    except AttributeError:
        raise TypeError("Le paramètre docs doit être une série pandas")



## Ajout du point en fin de ligne
#
# @param docs           Le corpus à traiter
# @return Le corpus où chaque ligne se termine par un point.
def add_point(docs):
    return docs.apply(lambda x: x+'.' if x[-1] is not '.' else x)


def repl(m):
    inner_word = m.group(2) + ' ' + m.group(3)
    return inner_word


## Ajout d'espaces avant et après les ponctuations
#
# @param docs           Le corpus à traiter
# @return               Le corpus traité
def deal_with_specific_characters(docs, pattern_after = r"(([A-Za-z])([',.;]))",
                                  pattern_before=r"(([',.;])([A-Za-z]))"):
    return docs.apply(lambda x: re.sub(pattern_before, repl, re.sub(pattern_after, repl, x)))

## Découpe un texte en tokens
#  Découpe un texte en sequences de maxlen tokens (word ou char) en max nbech
#
# @params text           [list<str>] Le texte à découper
# @params nbech          [int]       Le nombre maximal de séquences à générer
# @params seq_size       [int]       Le nombre de tokens par séquence
# @params step           [int]       Le recouvrement entre séquences
# @params granularity    [str]       word ou char: détermine le type de tokens présents dans text
# @returns sequences     [list<list<str>>] La liste de séquences générée
# @returns next_item     [str]       Pour chaque séquence de séquences, le token suivant
# TODO meilleure fusion possible du code (boucles for)
def split_text_into_tokens(text, nbech, seq_size, step, granularity="word"):
    compteur = 0
    sequences = []
    next_item = []

    # Cas des caractères
    if granularity == "char":
        for i in range(0, len(text) - seq_size, step):
            # Pour éviter de prendre des 'bouts de textes' composés de 2 phrases <>
            if "." in text[i: i + seq_size]:
                compteur += 0
            elif compteur < nbech:
                compteur += 1
                sequences.append(text[i: i + seq_size])
                next_item.append(text[i + seq_size])

    # Cas des mots
    elif granularity == "word":
        print("Granularity = by word")
        for sequence in text:

            for i in range(0, len(sequence) - seq_size, step):
                sequences.append(sequence[i: i + seq_size])
                next_item.append(sequence[i + seq_size])
            # Toujours ajouter la fin de la phrase pour bien détecter les fins de phrases
            if len(sequence) - seq_size - 1 % step != 0:
                sequences.append(sequence[len(sequence) - seq_size: len(sequence)])
                next_item.append(sequence[len(sequence) - 1])

    print('Construction d\'une base d\'apprentissage composée de :',
          len(sequences), 'séquences de longueur', seq_size)
    return sequences, next_item


## Vectorisation one-hot encoding de séquences
#  Transforme une list de n séquences de longueur seq_size de tokens ayant un vocabulaire de taille vocab_size
#  en matrice n x seq_size x vocab_size.
#
# @params sequences      [list<list<str>>] La liste de séquences à encoder
# @params seq_size       [int]             Le nombre de tokens par séquence
# @params vocab_size     [int]             Le nombre de tokens uniques (la taille du vocabulaire)
# @params tokens_indice  [dict(str,int)]   Dictionnaire de mapping token => index
# @params next_tokens    [list<str>]       Opt: pour chaque séquence, le token qui suit
# @returns x             [np.array]        La matrice n x seq_size x vocab_size
# @returns y             [np.array]        La matrice n x vocab_size des tokens suivants
def one_hot_vectorisation(sequences, seq_size, vocab_size, tokens_indice, next_tokens=None):
    x = np.zeros((len(sequences), seq_size, vocab_size), dtype=np.bool)
    if next_tokens:
        y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    else:
        y = None

    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            x[i, t, tokens_indice[char]] = 1
        if next_tokens:
            y[i, tokens_indice[next_tokens[i]]] = 1

    print('Vectorisation ... : ', len(sequence), "documents dupliqués en", seq_size, "vecteurs de", vocab_size,
          "colonnes")
    return (x, y)
