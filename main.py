import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from prediction import Prediction
from NextWordPred_db import Predictor_Database
from flask import Flask, request, render_template
from flask_cors import CORS
from flask_restful import reqparse
import pandas as pd
import pyttsx3
import speech_recognition as sr
import webbrowser as wb
import numpy as np
from image_recognition import openFile, detect_handwritten_ocr
import pickle
import heapq
import keras
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import sqlite3
import base64
import logging

#chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
# Speech recognizer


# https://pyttsx3.readthedocs.io/en/latest/engine.html
# TODO
# Speak instructions to user
# Speech to text and read out the 3 words reccomended or type out the first word


app = Flask(__name__)
cors = CORS(app)

prediction = Prediction()
pred_db = Predictor_Database('WORD_RECORD')
pred_db_login = Predictor_Database('LOGIN')
pred_db_feedback = Predictor_Database('FEEDBACK')

data_put_args = reqparse.RequestParser()
data_put_args.add_argument("word", type=str, help="Word Typed")
data_put_args.add_argument("isPredict", type=str,
                           help="Is Word Prrdicted (Y/N)")
data_get_args = reqparse.RequestParser()
data_get_args.add_argument("data", type=str, help="Word Typed")

# Render Application landing page
logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)


@app.route("/", methods=['POST','GET'])
def index():
    if(request.method == 'POST'):
        uname = request.form['uname']
        passwrd = request.form['pass']
        pred_db_login.insert(dict(EMAIL_ID=uname, PASSWORD=passwrd))
    textToSpeech("Welcome to Nextword AI")
    return render_template('index.html')

@app.route("/login", methods=['POST','GET'])
def login():
    if(request.method == 'POST'):
        uname = request.form['uname']
        passwrd = request.form['pass']
        pred_db_login.insert(dict(EMAIL_ID=uname, PASSWORD=passwrd))
    textToSpeech("Enter Email and password")
    return render_template('login.html')


@app.route('/main', methods=['POST'])
def main():
    print("reached main")
    template = ""
    url=""
    text=""
    uname = request.form['uname']
    passwrd = request.form['pass']
    data = pred_db_login.readWithWhere(dict(key='EMAIL_ID', value=uname))
    if(data != None):
        print("User exists")
        print(data[0])
        if(data[0]==uname and data[1]==passwrd):
            textToSpeech("Welcome to Next word prediction application")
            template = "main.html"
            if(uname == "sangeetha.nair9315@gmail.com" and passwrd == "123456"):
                url = "/viewFeedback"
                text = "View Feedback"
            else:
                url = "/collectFeedback"
                text = "Give Feedback"
        else:
            print("Invalid credentials. Please check your credentials and login again..")
            textToSpeech("Invalid credentials. Please check your credentials and login again")
            template = "login.html"
    else:
        print("User does not exist. Please sign up")
        textToSpeech("User does not exist. Please sign up")
        template = "signup.html"
    return render_template(template, url=url, feedback=text)


@app.route('/collectFeedback')
def collectFeedback():
    print("collectFeedback")
    textToSpeech("Please answer some of the questions")
    return render_template('feedback_form.html')

@app.route('/viewFeedback')
def viewFeedback():
    print("viewFeedback")
    plot_url1 = ""
    plot_url2 = ""
    plot_url3 = ""
    plot_url4 = ""
    template = ""
    cnx = sqlite3.connect('NextWordPredictor.db')
    cursor = cnx.execute('''SELECT * FROM FEEDBACK;''')
    dbdata = cursor.fetchone()
    if(dbdata != None):
        img1 = BytesIO()

        df = pd.read_sql_query("SELECT PREDICTION_ACCURACY, COUNT(PREDICTION_ACCURACY) as COUNT FROM FEEDBACK GROUP by PREDICTION_ACCURACY", cnx)
        data = dict(zip(df['PREDICTION_ACCURACY'].tolist(), df['COUNT'].tolist()))
        wc = WordCloud(width=800, height=400, max_words=200, background_color="white").generate_from_frequencies(data)
        plt.figure( figsize=(4,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img1, format='png')
        plt.close()
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')

        img2 = BytesIO()

        df = pd.read_sql_query("SELECT RECOMMEND, COUNT(RECOMMEND) as COUNT FROM FEEDBACK GROUP by RECOMMEND", cnx)
        df.plot.bar(x="RECOMMEND", y="COUNT", rot=70)
        plt.savefig(img2, format='png')
        plt.close()
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')

        img3 = BytesIO()

        df = pd.read_sql_query("SELECT HOW_APP, COUNT(HOW_APP) as COUNT FROM FEEDBACK GROUP by HOW_APP", cnx)
        data = dict(zip(df['HOW_APP'].tolist(), df['COUNT'].tolist()))
        wc = WordCloud(width=800, height=400, max_words=200, background_color="white").generate_from_frequencies(data)
        plt.figure( figsize=(4,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img3, format='png')
        plt.close()
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode('utf8')

        img4 = BytesIO()

        df = pd.read_sql_query("SELECT SPEECH_RECOGNITION, COUNT(SPEECH_RECOGNITION) as COUNT FROM FEEDBACK GROUP by SPEECH_RECOGNITION", cnx)
        colors = ['pink', 'silver', 'steelblue']
        plt.pie(df["COUNT"], labels = df["SPEECH_RECOGNITION"], colors = colors)
        plt.legend()
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)
        plot_url4 = base64.b64encode(img4.getvalue()).decode('utf8')
        textToSpeech("Please view the feedback of the application")
        template = "view_feedback.html"
    else:
        template = "no_feedback.html"
        textToSpeech("No feedback to be viewed at this time")
    return render_template(template, plot_url1=plot_url1 , plot_url2=plot_url2, plot_url3=plot_url3, plot_url4=plot_url4)


@app.route('/back')
def back():
    print("backmethod")
    return render_template('login.html')


@app.route('/signUp')
def signUp():
    print("Entered signUp")
    textToSpeech("Not a memebr. You need to create account")
    return render_template('signup.html')


@app.route('/GoodBye', methods=['POST'])
def home():
    print("GoodBye")
    pred_acc = request.form['Prediction_Accuracy']
    recommend = request.form['recommend']
    how_app = request.form['howapp']
    spch_recogn = request.form['Speech_Recognition']
    pred_db_feedback.insert(dict(PREDICTION_ACCURACY=pred_acc, RECOMMEND=recommend, HOW_APP=how_app, SPEECH_RECOGNITION=spch_recogn))
    textToSpeech("Your feedback has been noted")
    return render_template('ThankYou.html')


# API call to get user's speech
@app.get("/speech")
def getSpeech():
    wordSaid = speechToText()
    return {
        "wordSaid": wordSaid,
    }


@app.get("/image")
def getImage():
    file_name = openFile()
    if file_name:
        text = detect_handwritten_ocr(file_name)
        s = " ".join(text.split())
        return {
            "text" : s,
        }
    else:
        print("No file selected")


# API call to obtain next pridected word
@app.get("/prediction")
def get_next_words():
    data = request.args['data']
    predict_word = prediction.get_words_prediction(data)
    wordsToRead = getPredictions(predict_word)
    # Checks to see if there are words to be predicted
    if len(wordsToRead) == 0:
        textToSpeech(
            "Cannot Predict Word, Please keep typing and I will provide help.")
    else:
        textToSpeech("Your next predicted words are")
        for word in wordsToRead:
            textToSpeech(word)
    return predict_word


# API call to fetch data from Excel to be displayed on Web Application
@app.get("/excelData")
def get_excelData():
    data = pred_db.readAll()
    # print(data)
    return data


# API call to update Word into Excel
@app.put("/excelData")
def put_excelData():
    args = data_put_args.parse_args()
    word = args['word'].lower()
    print("Entered word :", word)
    isPredict = args['isPredict']
    word = format_word(word)
    print("Formatted word : '"+word+"'")
    pred_db.updInsertDb(dict(key='WORD', value=word), dict(
        WORD=word, COUNT=1, FLAG=isPredict), isPredict)
    data = pred_db.readAll()
    return data


# Function to convert speech to text
def speechToText():
    # Create a recognizer object
    r = sr.Recognizer()

    # Set the microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        # Record audio from the microphone
        audio = r.listen(source)
        print('Done!')

    # Recognize speech using Google Speech Recognition
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


# Function to run text to speech
def textToSpeech(word):
    engine = None
    engine = pyttsx3.init()
    engine.say(word)
    engine.runAndWait()
    # engine.iterate


# Function to adjust TTS volume
def changeVolume(change):
    engine = pyttsx3.init()
    volume = engine.getProperty('volume')
    if (change == 'up'):
        volume += 0.05
    else:
        volume -= 0.05
    engine.setProperty('volume', volume)

# Function get three words from payload


def getPredictions(data):
    wordsToPredict = []
    for entries in data:
        if (entries["accuracy"] != 0):
            wordsToPredict.append(entries["name"])
    return wordsToPredict


# Filter word to allow only alphanumeric character and '-' character
def format_word(word):
    if not word.isalnum():
        sample_list = []
        for i in word:
            if i.isalnum() or i == '-':
                sample_list.append(i)
        word = "".join(sample_list)
    return word.strip()

# Append new row to Excel

def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
