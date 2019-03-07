# -*- coding: utf-8 -*-

## Fonctions de gestion des stop words
# Auteurs : Task Force Sémantique
# Date : 10/07/2018
#
# Fonctions :
# - remove_accents
# - stopwords_ascii
# - stopwords_nltk
# - stopwords_nltk_ascii
# - default
# - remove_stopwords

import unicodedata
import nltk

# From R stopwords package (wrapper around https://github.com/stopwords-iso/stopwords-iso)
STOPWORDS = ["a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait",
             "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après",
             "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui", "aupres", "auquel",
             "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront",
             "aussi", "autre", "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient",
             "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons",
             "b", "bah", "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr",
             "c", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là",
             "celui", "celui-ci", "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines",
             "certains", "certes", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque",
             "cher", "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante",
             "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable", "comparables",
             "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de", "debout", "dedans", "dehors",
             "deja", "delà", "depuis", "dernier", "derniere", "derriere", "derrière", "des", "desormais", "desquelles",
             "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement", "devant", "devers", "devra",
             "devrait", "different", "differentes", "differents", "différent", "différente", "différentes",
             "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse", "diverses",
             "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc", "dont", "dos", "douze",
             "douzième", "dring", "droite", "du", "duquel", "durant", "dès", "début", "désormais", "e", "effet",
             "egale", "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin",
             "entre", "envers", "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues",
             "euh", "eurent", "eus", "eusse", "eussent", "eusses",  "eussiez", "eussions", "eut", "eux", "eux-mêmes",
             "exactement", "excepté", "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient",
             "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc",  "fois", "font", "force", "furent",
             "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gens",
             "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue",
             "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i", "ici", "il", "ils", "importe", "j", "je",
             "jusqu", "jusque", "juste", "k", "l", "la", "laisser", "laquelle", "las", "le", "lequel", "les",
             "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lors", "lorsque", "lui", "lui-meme", "lui-même",
             "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale", "me", "meme",
             "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "mine", "minimale", "moi",
             "moi-meme", "moi-même", "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même",
             "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire",
             "necessairement", "neuf", "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment",
             "notre", "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh",
             "ohé", "ollé", "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste",
             "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce", "parfois", "parle",
             "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier", "particulière",
             "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne", "personnes", "peu", "peut",
             "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce", "plein", "plouf", "plupart", "plus",
             "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour", "pourquoi",
             "pourrais", "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement",
             "pres", "probable", "probante", "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur",
             "pure", "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre",
             "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles",
             "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare",
             "rarement", "rares", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste",
             "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sait", "sans",
             "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble", "semblent",
             "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez",
             "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si", "sien", "sienne", "siennes",
             "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient", "sois", "soit", "soixante", "sommes",
             "son", "sont", "sous", "souvent", "soyez", "soyons", "specifique", "specifiques", "speculatif", "stop",
             "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit", "suivant", "suivante",
             "suivantes", "suivants", "suivre", "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis",
             "tant", "tardive", "te", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente",
             "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant",
             "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "trois",
             "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes",
             "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur", "vas", "vers", "via", "vif",
             "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà", "vont", "vos",
             "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut", "à", "â", "ça",
             "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions", "été", "étée", "étées", "étés",
             "êtes", "être", "ô"]

# Stopwords relatifs aux offres d'emploi PE.fr : mots très présents et peu porteurs de sens
STOPWORDS_OFFRES_1 = ["recherche", "recherchons", "mission", "missions", "poste", "recrute", "recrutons"]

# Stopwords relatifs aux offres d'emploi PE.fr : mots très présents mais pouvait être discriminants
STOPWORDS_OFFRES_2 = ["expérience", "assurer", "assurez", "travaux"]

## Enlève les accents
#
# Enlève les accents et autres caractères spéciaux (ç...) d'une chaine
# En conservant les caractères ASCII de base : ç devient c, ë devient e etc.
#
# @param str_input  [str] Chaine de texte dont il faut enlever les accents
# @return Chaine de texte sans accents
def remove_accents(text):
    return ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))


## Stopwords sans caractères spéciaux
#
# Renvoie la liste des stopwords au format ASCII, soit sans caractères spéciaux ni accents
# Appelle la fonction remove_accents du même package
#
# @return Liste des stopwords au format ASCII
def stopwords_ascii():
    return list(map(remove_accents, STOPWORDS))


## Renvoie les stopwords du package nltk
#
# Si demandé, met à jour le set de stopwords de nltk
# Renvoie le set de stopwords de nltk
#
# @param try_update [bool] Autorise la tentative d'update de la liste de stopwords (par défaut: False)
# @return Liste de stopwords français du package nltk
def stopwords_nltk(try_update=False):
    if try_update:
        nltk.download('stopwords')
    return nltk.corpus.stopwords.words('french')


## Stopwords nltk sans caractères spéciaux
#
# Renvoie la liste des stopwords nltk au format ASCII, soit sans caractères spéciaux ni accents
# Appelle la fonction remove_accents
#
# @return Liste des stopwords nltk au format ASCII
def stopwords_nltk_ascii():
    return list(map(remove_accents, stopwords_nltk()))


## Comportement par défaut
#
# Régit le fonctionnement par défaut du package stopwords
#
# @return Liste de stopwords
def default():
    print('Stopwords set not found. \nUsing default settings (available : iso, nltk and all)')
    return STOPWORDS


usage = {
    'iso': set().union(STOPWORDS, stopwords_ascii()),
    'nltk': set().union(stopwords_nltk(try_update=True), stopwords_nltk_ascii()),
    'offres_pe': set().union(STOPWORDS_OFFRES_1, STOPWORDS_OFFRES_2),
    'all': set().union(STOPWORDS, stopwords_ascii(), stopwords_nltk(), stopwords_nltk_ascii(), STOPWORDS_OFFRES_1,
                       STOPWORDS_OFFRES_2)
}


## Supprime les stopwords
#
# Supprime les stopwords d'un texte donné et renvoie le texte dans stopwords
# L'ensemble des stopwords peut être défini via le paramètre "option"
# Ce paramètre peut prendre les valeurs "iso", "nltk", "all".
#
# @param str_input  [str] Texte dont on veut enlever les stopwords
# @param option     [str] Définit l'ensemble de stopwords à utiliser
# @return Texte dans stopwords
def remove_stopwords(str_input, opt='all'):
    if opt in usage.keys():
        stopwords_set = usage.get(opt)
    else:
        stopwords_set = default()
    return [c for c in str_input if c not in stopwords_set]
