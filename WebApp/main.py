#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter

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
    flash('You will have 45 seconds to discuss the topic mentioned above. Due to restrictions, we are not able to redirect you once the video is over. Please move your URL to /dash instead of /video_1 once over. You will be able to see your results then.')
    return render_template('video.html')

@app.route('/video_1', methods=['POST'])
def video_1() :
    try :
        return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(stream_template('video.html', gen()))
    except :
        return None

@app.route('/dash', methods=("POST", "GET"))
def dash():
    df_2 = pd.read_csv('static/js/histo_perso.txt')


    def emo_prop(df_2) :
        return [int(100*len(df_2[df_2.density==0])/len(df_2)),
                    int(100*len(df_2[df_2.density==1])/len(df_2)),
                    int(100*len(df_2[df_2.density==2])/len(df_2)),
                    int(100*len(df_2[df_2.density==3])/len(df_2)),
                    int(100*len(df_2[df_2.density==4])/len(df_2)),
                    int(100*len(df_2[df_2.density==5])/len(df_2)),
                    int(100*len(df_2[df_2.density==6])/len(df_2))]

    emotions = ["Angry", "Disgust", "Fear",  "Happy", "Sad", "Surprise", "Neutral"]
    emo_perso = {}
    emo_glob = {}

    for i in range(len(emotions)) :
        emo_perso[emotions[i]] = len(df_2[df_2.density==i])
        emo_glob[emotions[i]] = len(df[df.density==i])

    df_perso = pd.DataFrame.from_dict(emo_perso, orient='index')
    df_perso = df_perso.reset_index()
    df_perso.columns = ['EMOTION', 'VALUE']
    df_perso.to_csv('static/js/hist_vid_perso.txt', sep=",", index=False)

    df_glob = pd.DataFrame.from_dict(emo_glob, orient='index')
    df_glob = df_glob.reset_index()
    df_glob.columns = ['EMOTION', 'VALUE']
    df_glob.to_csv('static/js/hist_vid_glob.txt', sep=",", index=False)

    emotion = df_2.density.mode()[0]
    emotion_other = df.density.mode()[0]

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

    return render_template('dash.html', emo=emotion_label(emotion), emo_other = emotion_label(emotion_other), prob = emo_prop(df_2), prob_other = emo_prop(df))


################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():
    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)


# Audio Emotion Analysis
@app.route('/audio_analysis', methods=("POST", "GET"))
def audio_analysis():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static","js", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static","js", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static', 'js','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static","js", "audio_emotions_other.txt"), sep=",")

    # Get most common emotion during the interview for other candidates
    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static', 'js','audio_emotions_dist_other.txt'), sep=',')

    # Sleep
    time.sleep(0.5)

    return render_template('audio_analysis.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other)


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

    counts = Counter(words)

    num_words = len(text.split())
    return common_words, num_words, counts

@app.route('/text_1', methods=['POST'])
def text_1():

    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    df_text = pd.read_csv('static/js/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/text.txt', sep=",", index=False)

    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]

    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']

    df_text_perso.to_csv('static/js/text_perso.txt', sep=',', index=False)

    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])

    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]

    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']

    df_mean.to_csv('static/js/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']

    probas = [int(e*100) for e in probas]

    data_traits = zip(traits, probas)

    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []

    common_words, num_words, counts = get_text_info(text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    with open("static/js/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    with open("static/js/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('result.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/text_pdf', methods=['POST'])
def text_pdf():
    f = request.files['file']
    f.save(secure_filename(f.filename))

    text = parser.from_file(f.filename)['content']
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()

    df_text = pd.read_csv('static/js/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/text.txt', sep=",", index=False)

    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]

    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']

    df_text_perso.to_csv('static/js/text_perso.txt', sep=',', index=False)

    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])

    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]

    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']

    df_mean.to_csv('static/js/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']

    probas = [int(e*100) for e in probas]

    data_traits = zip(traits, probas)

    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []

    common_words, num_words, counts = get_text_info(text)

    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)

    trait = traits[probas.index(max(probas))]

    with open("static/js/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    with open("static/js/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('result.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

if __name__ == '__main__':
    app.run(debug=True)
