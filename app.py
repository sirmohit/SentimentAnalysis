from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils import pad_sequences
import tensorflow as tf
import keras





app = Flask(__name__)



def init():
    global model,graph
    # load the pre-trained Keras model
    # model = load_model('sentimental.h5')
    graph = tf.compat.v1.get_default_graph()

#########################Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("index.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        text = request.form['text']
        Sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split() #split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        x_test = keras.utils.pad_sequences(x_test, maxlen=300) # Should be same which you used for training data
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            model = load_model('sentimental.h5')
            probability = model.predict(array([vector][0]))[0][0]
            class1 = np.argmax(model.predict(array([vector][0]))[0][0])
        if probability <0.5:
            sentiment = "NEGATIVE 🙁🙁🙁" 
        
        else:
            sentiment = 'POSITIVE 😊😊😊'
            
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability)
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.debug = False
    app.run(host='0.0.0.0', port=8080)
