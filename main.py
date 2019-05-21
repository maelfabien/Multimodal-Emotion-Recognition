#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, session, request, redirect, flash
from live_face import show_webcam
import pandas as pd

app = Flask(__name__)

app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'

global val

@app.route('/rules', methods=("POST", "GET"))
def html_table():
    
    return render_template('histo.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/newgame', methods=['GET', 'POST'])
def newgame() :
    #if request.method == 'POST':
    session['score'] = 0
    
    if 'start_auto' in request.form :
        session['article'] = request.form['start']
        return redirect('/autogame')
    else :
        session['article'] = request.form['start']
        return redirect('/game')


@app.route('/video', methods=['POST'])
def move() :
    show_webcam(session['article'])


@app.route('/rules')
def tuto():
    return render_template('rules.html')

# Si vous définissez de nouvelles routes, faites-le ici

if __name__ == '__main__':
    app.run(debug=True)
