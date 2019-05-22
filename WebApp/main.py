#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division

import numpy as np
import pandas as pd
import cv2

import time
from time import sleep
import re
import os

import argparse
from collections import OrderedDict

### Audio Library ###
from speechEmotionRecognition import *

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib

from tensorflow.keras.models import load_model
from imutils import face_utils
import requests

from flask import Flask, render_template, session, request, redirect, flash, Response
import pandas as pd
import cv2
import sys

from live_face import gen
from predict import *
from nltk import *

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

app = Flask(__name__)

app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'

global val

df = pd.read_csv('histo.txt', sep=",")

plt.figure(figsize=(12,8))
plt.hist(df.values.flatten(), bins=20, density=True)
plt.savefig('static/CSS/new_plot.png')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/dash', methods=("POST", "GET"))
def html_table():

    return render_template('dash.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/video', methods=['POST'])
def video() :

    try :
        return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    except :
        return redirect('/')


def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):

    words = tokenize.word_tokenize(text)
    common_words = FreqDist(words).most_common(100)

    # Retrieve some info on the text data
    print(common_words)
    print(str(' '.join([e[0] for e in common_words])))

    wordcloud = WordCloud().generate(str(' '.join([e[0] for e in common_words])))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/CSS/wordcloud.png')

    num_words = len(text.split())
    return common_words, num_words


@app.route('/text', methods=['POST'])
def text():
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    probas = [np.round(e,2) for e in probas]

    data_traits = zip(traits, probas)

    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []

    common_words, num_words = get_text_info(text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    plt.figure()
    # Example data
    y_pos = np.arange(len(traits))
    plt.bar(y_pos, probas, align='center')
    #plt.gca().set_yticks(y_pos)
    plt.gca().set_xticklabels(traits)
    #plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.xlabel('Probability')
    plt.title('Sentiment')

    plt.savefig('static/CSS/sentiment.png')

    #flash('Your dominant personality trait is : {}'.format(str(trait)))
    return render_template('result.html', traits = data_traits, trait = trait, num_words = num_words, common_words = common_words)


@app.route('/audio', methods=['POST'])
def audio() :

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Recording
    recording_sub_dir = os.path.join('audio', 'voice_recording.wav')
    SER.voice_recording(recording_sub_dir)

    # Predict emotion in voice
    emotions = SER.predict_emotion_from_file(recording_sub_dir)
    print('Predicted emotions:')
    print(emotions)

    return redirect('/')


@app.route('/rules')
def rules():
    return render_template('rules.html')

#if cv2.waitKey(1) & 0xFF == ord('q'):
#break

#video_capture.release()
#cv2.destroyAllWindows()
#sys.modules[__name__].__dict__.clear()


if __name__ == '__main__':
    app.run(debug=True)
