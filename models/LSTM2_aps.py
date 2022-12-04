import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import keras.utils
from keras.callbacks import EarlyStopping
#Necessary imports for our model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from keras import backend as K
import seaborn as sns
from sklearn.metrics import confusion_matrix


#nltk.download('stopwords')
# for Colab users: !pip install tensorflow_text
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

#load the data
df = pd.read_csv('/home/apostolos/Downloads/seng544-proj-2-travis/ML Analysis/data.csv')

##########################
### Text Preprocessing ###
##########################

#remove rt from begining of sentence - do first cause RT is capitalized.
df["text"] = df["text"].map(lambda name: re.sub('^(RT)', ' ', name))
#removing links
df["text"] = df["text"].map(lambda name: re.sub(r'http\S+', ' ', name))
#removing mentions
df["text"] = df["text"].map(lambda name: re.sub("@([a-zA-Z0-9_]{1,50})", '', name))

#remove repeated instances of characters
#removing repeating characters
repeat_pattern = re.compile(r'(\w)\1*') #compile the pattern we are looking for
match_substitution = r'\1' #substituion pattern
df["text"] = df["text"].map(lambda name: re.sub(repeat_pattern, match_substitution, name))
#removal of digits with regex - we do this here because it is possible to have numbers in tags and urls replace with space.
df["text"] = df["text"].map(lambda name: re.sub(r'[0-9]', ' ', name))
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+")
df["text"] = df["text"].map(lambda name: re.sub(EMOJI_PATTERN, ' ', name))
#do this after removing mentions -> don't # here. ->replace with space.
df["text"] = df["text"].map(lambda name: name.lower())
special_pattern = re.compile('[!\.\^\$\|\?\*\+\=\(\)\{\}\@\=\/\<\>\,\~\`\-\%\&\:\;\[\]"“”…]')
df["text"] = df["text"].map(lambda name: re.sub(special_pattern, ' ', name))
#remove a hashtag if it has no significance, ie, not part of a #word
df["text"] = df["text"].map(lambda name: re.sub('(#[^(a-zA-Z0-9)])', ' ', name))
#removing doublicate spaces and all white spaces like \t, \n or \r
df["text"] = df["text"].map(lambda name: " ".join(name.split()))
#Now remove stop words
df["text"] = df["text"].map(lambda name: ' '.join([word for word in name.split() if word not in stopwords_dict]))
#After removing stop words we can clean up more
df["text"] = df["text"].map(lambda name: re.sub('[\']', ' ', name))
#final white space clean up
df["text"] = df["text"].map(lambda name: " ".join(name.split(' ')))
#still need to check for strings that contain whitespaces only and remove them
df["text"] = df["text"].map(lambda text: np.nan if len(text) == 0 else text)
df.dropna(axis=0, inplace=True)
df.to_csv("testing.csv", sep='\t', encoding='utf-8')

##############################
### End Text Preprocessing ###
##############################

#Select number of countries that we want our model to examine (Top in # of tweets)
num_of_top_countries = 20
df = df[df["country"].isin(df["country"].value_counts()[:num_of_top_countries].index.values)]

plt.style.use('ggplot')

#visualization of the # of tweets per country aftter text preprocessing
num_classes = len(df["country"].value_counts())
colors = plt.cm.Dark2(np.linspace(0, 1, num_classes))
iter_color = iter(colors)

df["country"].value_counts().plot.barh(title="Tweet for each country (n, %)", 
                                       ylabel="Countries", 
                                       color=colors,figsize=(29,29))

for i, v in enumerate(df["country"].value_counts()):
    c = next(iter_color)
    plt.text(v, i,
             " "+str(v)+", "+str(round(v*100/df.shape[0], 2))+ "%",
             color=c,
             va='center',
             fontweight='bold')
plt.show()


#map countries to integers
#create dictionary
country_dict = dict()
countries = df["country"].unique()
for i in range(0, num_classes):
    country_dict[countries[i]] = i

df["Labels"] = df["country"].map(country_dict)

#drop country, we are interested in the numerical mapping of the label only.
df = df.drop(["country"], axis=1)

X = df["text"]
y = tf.keras.utils.to_categorical(df["Labels"].values, num_classes=num_classes)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0, stratify=y)
X_train,X_valid,y_train,y_valid = train_test_split(X_train, y_train,test_size=0.1, random_state=0, stratify=y_train)


max_words = 1000
max_len = 250
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_len)
tok.fit_on_texts(X_valid)
sequences_valid = tok.texts_to_sequences(X_valid)
sequences_matrix_valid = tf.keras.preprocessing.sequence.pad_sequences(sequences_valid ,maxlen=max_len)



def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,100,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(num_classes,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#Create and train a classification model

#Definition of functions that will be used to calculate metrics
def balanced_recall(y_true, y_pred):
    #This function calculates the balanced recall metric
    #recall = TP / (TP + FN)

    recall_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recall_by_class = recall_by_class + recall
    return recall_by_class / y_pred.shape[1]

def balanced_precision(y_true, y_pred):
    #This function calculates the balanced precision metric
    #precision = TP / (TP + FP)

    precision_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        precision_by_class = precision_by_class + precision
    # return average balanced metric for each class
    return precision_by_class / y_pred.shape[1]

def balanced_f1_score(y_true, y_pred):
    #This function calculates the F1 score metric
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#define prediction function, predict class of input tweets
#Arguments : tweets (list of strings)
#Returns   : class (list of int where the tweets belong)
def predict_class(tweets):
  #pick the class for which the highest probability the tweet originated from
  return [np.argmax(pred) for pred in model.predict(tweets)]

METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
      balanced_recall,
      balanced_precision,
      balanced_f1_score
]

model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=METRICS)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)


model_fit = model.fit(sequences_matrix,y_train,batch_size=32,epochs=10,
          validation_data = (sequences_matrix_valid, y_valid), callbacks=[earlystop_callback])



test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,maxlen=max_len)



accr = model.evaluate(test_sequences_matrix,y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

y_pred = predict_class(test_sequences_matrix)
y_pred_categorical = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes)
print(classification_report(y_test, y_pred_categorical))

#plot the values assumed by each monitored metric during training procedure. Compare training/validation curves
metric_list = list(model_fit.history.keys())
num_metrics = int(len(metric_list)/2)
x = list(range(1, len(model_fit.history['loss'])+1))


fig, ax = plt.subplots(nrows=1, ncols=num_metrics, figsize=(30, 5))

for i in range(0, num_metrics):
    ax[i].plot(x, model_fit.history[metric_list[i]], marker="o", label=metric_list[i].replace("_", " "))
    ax[i].plot(x, model_fit.history[metric_list[i+num_metrics]], marker="o", label=metric_list[i+num_metrics].replace("_", " "))
    ax[i].set_xlabel("epochs",fontsize=14)
    ax[i].set_title(metric_list[i].replace("_", " "),fontsize=20)
    ax[i].legend(loc="lower left")
plt.show()


#Confusion Matrix to be added
# Convert predictions classes to one hot vectors 
# Predict the values from the validation dataset
Y_pred = model.predict(test_sequences_matrix)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
g = sns.heatmap(confusion_mtx, annot=True, xticklabels=countries, yticklabels=countries, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
g.set_yticklabels(g.get_yticklabels(), rotation = 0)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
