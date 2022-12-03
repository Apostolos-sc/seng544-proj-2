import pandas as pd
import nltk
import random
import numpy as np
import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy
import tkinter
from tkinter import *

# Import data
data = pd.read_csv("../Data/cleaned_tweets.csv")

print(data.iloc[10]['text'])

'''
# Initialize
lemmatizer = nltk.stem.WordNetLemmatizer()
words = [] # List of all words used in any tweet
classes = [] # List of all countries
documents = [] # List of tuples of words and countries

# Preprocess words
data['text'] = data['text'].apply(lambda x: str(x))

# Iterate through tweets and classify tokens by country
print("Tokenizing...")
for n in range(0, len(data)):
    
    # Take each tweet and tokenize it then add the tokens to a list
    w = nltk.word_tokenize(data.iloc[n]['text'])
    words.extend(w)

    # Add documents
    documents.append((w, data.iloc[n]['country']))

    # Add classes
    if data.iloc[n]['country'] not in classes:
        classes.append(data.iloc[n]['country'])

# Lemmatize words
words = [lemmatizer.lemmatize(w.lower()) for w in words]

# Sort lists
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Initialize training data
training = []
output_empty = [0] * len(classes)

print("Vectorizing...")
for doc in documents:

    # Initialize bag of words
    bag = []

    # List of tokenized words for the pattern
    pattern_words = doc[0]

    # Lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create our bag of words array with 1 if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a 0 for each tag and 1 for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
'''

X = CountVectorizer().fit_transform(data['text'])
Y = LabelEncoder().fit_transform(data['country'])
Y = keras.utils.np_utils.to_categorical(Y)
print(Y.shape)

# Split features
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
scipy.sparse.csr_matrix.sort_indices(X_train)
scipy.sparse.csr_matrix.sort_indices(X_test)

# Create train and test lists
#X_train = list(training[:,0])
#Y_train = list(training[:,1])

# Create model
print("Building model...")
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(Y_train.shape[1], activation='softmax'))

# Compile model
print("Compiling model...")
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model
print("Fitting model...")
print(type(X_train))
print(type(Y_train))
print(type(X_train))
hist = model.fit(X_train, Y_train, epochs=2, batch_size=16, verbose=1)
model.save('chatbot.h5', hist)

print("Predicting...")
print(model.predict(X_test))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    #sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w ==s:
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return np.array(bag)

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    #p = bow(sentence, words, show_details=False)
    #res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    #results = [[i, r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Sort by strength of probability
    #results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    #for r in results:
        #return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = "hello"#getResponse(ints, intents)
    return res

def send():
    print("send() called.")
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=False, height=False)

# Create chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# Bind scrollbar to chat windows
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create button to send messages
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height="5", bd=0, bg="#32de97", activebackground="#3c9d9b", fg="#ffffff", command=send)

# Create the box to enter messages
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()