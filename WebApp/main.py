#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response

### Audio imports ###
from speechEmotionRecognition import *

### Video imports ###
from live_face import *

### Text imports ###
from predict import *
from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile


# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

################################################################################
################################## INDEX #######################################
################################################################################

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

################################################################################
################################## RULES #######################################
################################################################################

@app.route('/rules')
def rules():
    return render_template('rules.html')

################################################################################
############################### VIDEO INTERVIEW ################################
################################################################################

df = pd.read_csv('static/js/histo.txt', sep=",")

@app.route('/video', methods=['POST'])
def video() :
    return render_template('video.html')

@app.route('/video_1', methods=['POST'])
def video_1() :
    try :
        return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    except :
        return None

@app.route('/dash', methods=("POST", "GET"))
def dash():
    df_2 = pd.read_csv('static/js/histo_perso.txt')
    emotion = df_2.density.mode()[0]

    def emo_prop(df_2) :
        return [int(100*len(df_2[df_2.density==0])/len(df_2)),
                    int(100*len(df_2[df_2.density==1])/len(df_2)),
                    int(100*len(df_2[df_2.density==2])/len(df_2)),
                    int(100*len(df_2[df_2.density==3])/len(df_2)),
                    int(100*len(df_2[df_2.density==4])/len(df_2)),
                    int(100*len(df_2[df_2.density==5])/len(df_2)),
                    int(100*len(df_2[df_2.density==6])/len(df_2))]

    def emotion_label(emotion) :
        if emotion == 0 :
            return "Angry"
        elif emotion == 1 :
            return "Disgust"
        elif emotion == 2 :
            return "Fear"
        elif emotion == 3 :
            return "Happy"
        elif emotion == 4 :
            return "Sad"
        elif emotion == 5 :
            return "Surprise"
        else :
            return "Neutral"

    return render_template('dash.html', emo=emotion_label(emotion), prob=emo_prop(df_2))


################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

 # Audio Global Variable
 df_audio_hist = pd.read_csv(os.path.join("static","js", "audio_emotions_hist.txt"), sep=",")

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():
    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Recording
    rec_duration = 10 # in sec
    rec_sub_dir = os.path.join('voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Add results to flask session
    session['speech_emotions'] = emotions
    session['speech_timestamp'] = timestamp

    # Export results to txt format
    SER.prediction_to_csv(self, emotions, os.path.join("static","js", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(self, emotions, os.path.join("static","js", "audio_emotions_hist.txt"), mode='a')

    # Send Flash message
    flash('The recording is over. You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.')

    return render_template('audio.html', display_button=True)


# Audio Emotion Analysis
@app.route('/audio_analysis', methods=("POST", "GET"))
def audio_analysis():

    # Get all audio emotions during the interview
    df = pd.read_csv(os.path.join("static", "js", "audio_emotions.txt"))

    # Get most common emotion during the interview
    most_common_emotion = df.EMOTIONS.mode()[0]

    # Calculate emotion distribution
    emotion_distribution = int(100 * df.EMOTIONS.value_counts() / len(df))
    print (emotion_distribution)

    return render_template('audio_analysis.html', emo=most_common_emotion, prob=emotion_distribution)


################################################################################
############################### TEXT INTERVIEW #################################
################################################################################

global df_text

tempdirectory = tempfile.gettempdir()

@app.route('/text', methods=['POST'])
def text() :
    return render_template('text.html')

def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):

    words = tokenize.word_tokenize(text)
    common_words = FreqDist(words).most_common(100)

    num_words = len(text.split())
    return common_words, num_words

@app.route('/text_1', methods=['POST'])
def text_1():
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    print(pd.DataFrame([probas], columns=traits))

    df_text = pd.read_csv('static/js/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/text.txt', sep=",", index=False)

    df_text_perso = pd.read_csv('static/js/text.txt', sep=",")
    df_text_perso = pd.DataFrame([probas], columns=traits)
    df_text_perso.to_csv('static/js/text_perso.txt', sep=',', index=False)

    print(np.mean(df_new['Extraversion']))
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    print(means)
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    print(df_mean)
    df_mean.to_csv('static/js/text_mean.txt', sep=',', index=False)

    probas = [int(e*100) for e in probas]

    data_traits = zip(traits, probas)

    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []

    common_words, num_words = get_text_info(text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    return render_template('result.html', traits = data_traits, trait = trait, num_words = num_words, common_words = common_words)

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/text_pdf', methods=['POST'])
def text_pdf():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    print(f)
    print(f.filename)

    text = parser.from_file(f.filename)['content']
    print(text)

    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    print(pd.DataFrame([probas], columns=traits))

    df_text = pd.read_csv('static/js/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/text.txt', sep=",", index=False)

    df_text_perso = pd.read_csv('static/js/text.txt', sep=",")
    df_text_perso = pd.DataFrame([probas], columns=traits)
    df_text_perso.to_csv('static/js/text_perso.txt', sep=',', index=False)

    probas = [int(e*100) for e in probas]

    data_traits = zip(traits, probas)

    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []

    common_words, num_words = get_text_info(text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]
    os.remove(f.filename)
    return render_template('result.html', traits = data_traits, trait = trait, num_words = num_words, common_words = common_words)





if __name__ == '__main__':
    app.run(debug=True)
