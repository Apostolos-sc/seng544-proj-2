import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WorldNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WorldNetLemmatizer()

words = []
classes = []
documents = []
ignore_words=['?', '!']
data_file = open('intents.json').read()
intents= json.load(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # Take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # Add documents
        documents.append((w, intent['tag']))

        # Add classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])