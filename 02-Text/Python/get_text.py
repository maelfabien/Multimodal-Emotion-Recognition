#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from bs4 import BeautifulSoup
from json import loads
from urllib.request import urlopen
from urllib.parse import urlencode
import ssl
import re
import urllib
from flask import Flask, render_template, session, request, redirect, flash, url_for
from predict import *
from nltk import *


def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):
    # Retrieve some info on the text data
    words = tokenize.word_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    num_words = len(text.split())
    return common_words, num_words

app = Flask(__name__)
app.secret_key = "motdepasse"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/text', methods=['POST'])
def text():
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    data_traits = zip(traits, probas)
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    common_words, num_words = get_text_info(text)
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    trait = traits[probas.index(max(probas))]
    #flash('Your dominant personality trait is : {}'.format(str(trait)))
    return render_template('result.html', traits = data_traits, trait = trait, num_words = num_words, common_words = common_words)


if __name__ == '__main__':
    app.run(debug=True)


