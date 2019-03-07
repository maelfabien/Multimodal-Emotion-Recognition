# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:43:50 2018

@author: irbr3490
"""

import pandas as pd
import re

"""df = pd.read_csv("C:/Users/IRBR3490/Desktop/PREPROCESSING/OffresWord2JOB_test.csv", sep = ",",encoding = "UTF-8")
df.head()
docs=df["dc_descriptifoffre"]

docs_notnull=notnull(docs)
docs_preprocess=preprocess_fundation(docs_notnull) 
docs_preprocess[1]

d=malefemale_listing()
liste1=d.listing_synonym(docs_preprocess)
liste1"""


#---------------------------------------------------------------------------------------------------------------------------
#Remplacement des mots masculin/féminin par le mot masculin
#---------------------------------------------------------------------------------------------------------------------------
#Objectif : effectuer le remplacement, dans le corpus de textes, des synonymes masculins/féminins par le seul masculin

#Deux classes : 
#CLASS malefemale_listing : donne la liste des synonymes
# @param docs
# @return synonym_set : listing des synonymes masculins, féminins et de leur correspondance finale (ex : (serveur,serveuse,serveur))
#CLASS replace_synonym : retourne le document avec suppression des synonymes
# @param docs
# @return docs_replace : retourne le document avec suppression des synonymes

class malefemale_listing :
    
    def __init__(self) :
    
        #Listing des correspondances fin mot1/fin mot2 pour identification des synonymes masculin/féminin
        self.listing = dict()
        self.listing["eur"] = ('euse','se','rice','e','eure')
        self.listing["ien"] = ('ne','nne','e')
        self.listing["ier"] = ('ere','re','e')
        self.listing["ial"] = ('e')
        self.listing["l"] = ('le')
        self.listing["nt"] = ('e')
        self.listing["at"] = ('e')
        self.listing["f"] = ('ve')
        self.listing["o"] = ('a')
        self.listing["er"] = ('ere','ère','euse')
        self.listing["i"] = ('e')
        self.listing["d"] = ('e')
        self.listing["é"] = ('e')
        self.listing["e"] = ('e')
        self.listing["on"] = ('onne')
        self.listing["is"] = ('ise')
        self.listing["s"] = ('s')        

    def preprocess (self,docs): #Formatage des textes (passage en minuscules, suppression des accents...)
        
        self.docs_preprocess=docs

        #Suppression des espaces éventuels entourant le /
        self.docs_preprocess= self.docs_preprocess.str.replace('(\s*)/(\s*)','/')
        #Suppression des espaces éventuels précédant le (
        self.docs_preprocess= self.docs_preprocess.str.replace('(\s*)\((\s*)','(')
        #Suppression des espaces éventuels après le )
        self.docs_preprocess= self.docs_preprocess.str.replace('\)',') ')   

        return  self.docs_preprocess
      
    def matching_words (self,word1, word2): #Pour un couple de mots word1,word2 donné : on restitue (word1,word2,word1) si word1, word2 sont des synonymes masculins/féminins,
                                            #(word1,word2,"unknown") sinon
        
    #On définit la correspondance entre les mots si :
    # Pour un word1_end donné et un word2_end donné possible :
    #   word2==word2_end   Exemple : serveur(se) ; serveur/se
    #   word2[-len(word2_end):]==word2_end and word1[:3]==word2[:3]  Exemple : serveur(serveuse) ; serveur/serveuse
    #   word2[-len(word2_end):]==word2_end and len(word2)<=(len(word2_end)+2)  Exemple : conducteur(trice) : en effet
    #                                                        'rice' est dans la liste des word2_end de 'teur' mais pas 'trice'

        combi = None
        for word1_end in self.listing.keys() :
            for word2_end in self.listing[word1_end] :
                if combi==None : 
                        if (word1[-len(word1_end):]==word1_end) :
                                if (word2==word2_end) or (word2[-len(word2_end):]==word2_end and word1[:4]==word2[:4]) or (word2[-len(word2_end):]==word2_end and len(word2)<=(len(word2_end)+2)):
                                    combi=(word1,word2,word1)
        if combi == None:
            combi = (word1,word2,"unknown")
        return combi
    
    def synonym (self,match,numligne): #Ajout au listing des synonymes les synonymes masculin/féminin trouvés pour un match donné

        combi=None

        if match != []:
            for (word1,word2,word3) in match :
                if word1==word2 and re.match ("^[\d\s]+$",word1) is None :    # EXEMPLE : Cas de serveur/serveur(se)
                    combi=(word1,word2,word1)
                else :
                    combi=self.matching_words(word1,word2)
                if combi is not None :
                    if (list(combi)[2]) != "unknown" and combi not in self.synonym_set.keys() : 
                        self.synonym_set[combi] = numligne

    def listing_synonym(self,docs): #Définition de la liste des synonymes masculin/féminin sur la base de l'ensemble du corpus
        
        self.preprocess(docs)
        
        self.synonym_set={}
        
        match_pattern=list()
        match_pattern.append("([\w\-]+)\(([\w\-]+)\)()") #Cas de serveur(se)
        match_pattern.append("([\w\-]+)/([\w\-]+)(\([\w\-]+\))?") #Cas de serveur/serveuse et serveur/serveur(se)
        match_pattern.append("([\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+)()") #Cas de apprenti boucher/apprentie bouchere
        match_pattern.append("([\w\-]+\s[\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+\s[\w\-]+)()") #Cas de aide apprenti boucher/aide apprentie bouchere
        
        for i in range(len(self.docs_preprocess)) : #Pour chaque texte du document et pour chaque match, on applique les fonctions précédentes
            for match in match_pattern : 
                self.synonym(re.findall(match,self.docs_preprocess[i]),i)
 
        return self.synonym_set 

class replace_synonym :

    def preprocess (self,docs): #Formatage des textes (passage en minuscules, suppression des accents...)
        self.docs_preprocess=docs
        #Suppression des espaces éventuels entourant le /
        self.docs_preprocess= self.docs_preprocess.str.replace('(\s*)/(\s*)','/')
        #Suppression des espaces éventuels précédant le (
        self.docs_preprocess= self.docs_preprocess.str.replace('(\s*)\((\s*)','(')
        #Suppression des espaces éventuels après le )
        self.docs_preprocess= self.docs_preprocess.str.replace('\)',') ')
        return  self.docs_preprocess

    def text_replace_synonym (self,text,synonym_set) : #Pour un texte donné et pour plusieurs cas de matching, 
                                                       #remplace les synonymes musculin/féminin par le mot masculin
        #Cas de serveur(se)
        parenthesis_pattern = re.compile(r"([\w\-]+)\(([\w\-]+)\)()")                                         
        match_parenthesis_pattern=parenthesis_pattern.findall(text)
        #Cas de serveur/serveuse et serveur/serveur(se)
        slash_pattern = re.compile(r"([\w\-]+)/([\w\-]+)(\([\w\-]+\))?")                                               
        match_slash_pattern=slash_pattern.findall(text)
        #Cas de apprenti boucher/apprentie bouchere
        slash_pattern_BiWords=re.compile(r"([\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+)()")                                          
        match_slash_pattern_BiWords=slash_pattern_BiWords.findall(text)        
        #Cas de aide apprenti boucher/aide apprentie bouchere
        slash_pattern_TriWords=re.compile(r"([\w\-]+\s[\w\-]+\s[\w\-]+)/([\w\-]+\s[\w\-]+\s[\w\-]+)()")                                        
        match_slash_pattern_TriWords=slash_pattern_TriWords.findall(text)

        if match_slash_pattern != []:
            for (mot1,mot2,mot3) in match_slash_pattern :
                    if mot1==mot2 :    # Cas de serveur/serveur(se)
                        text=text.replace(mot1+"/"+mot2+"("+mot3+")",mot1)
                        text=text.replace(mot1+"/"+mot2,mot1)
                    elif (mot1,mot2,mot1) in synonym_set :
                        text=text.replace(mot1+"/"+mot2+"("+mot3+")",mot1)
                        text=text.replace(mot1+"/"+mot2,mot1)

        if match_parenthesis_pattern != []:
            for (mot1,mot2,mot3) in match_parenthesis_pattern :
                if (mot1,mot2,mot1) in synonym_set :
                    text=text.replace(mot1+"("+mot2+")",mot1)

        if match_slash_pattern_BiWords != []:
            for (mot1,mot2,mot3) in match_slash_pattern_BiWords :
                if (mot1,mot2,mot1) in synonym_set :
                    text=text.replace(mot1+"/"+mot2,mot1)

        if match_slash_pattern_TriWords != []:
            for (mot1,mot2,mot3) in match_slash_pattern_TriWords :
                if (mot1,mot2,mot1) in synonym_set :
                    text=text.replace(mot1+"/"+mot2,mot1)
        return text 

    def doc_replace_synonym (self,docs,synonym_set) :
        docs_replace=[]
        self.preprocess(docs)
        for i in (range(len(self.docs_preprocess))) :
            docs_replace.append(self.text_replace_synonym(self.docs_preprocess[i],synonym_set))

        return pd.Series(docs_replace)



