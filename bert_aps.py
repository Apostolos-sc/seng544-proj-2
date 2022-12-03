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


#nltk.download('stopwords')
# for Colab users: !pip install tensorflow_text
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

#load the data
df = pd.read_csv('data.csv')

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

#Create training, validation, test set
y = tf.keras.utils.to_categorical(df["Labels"][:10000].values, num_classes=num_classes)
X_train, X_test, y_train, y_test = train_test_split(df["text"][:10000], y, test_size=0.2, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state=0)

#BERT layers
preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")
#preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
#encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_embeddings(sentences):
  '''return BERT-like embeddings of input text
  Args:
    - sentences: list of strings
  Output:
    - BERT-like embeddings: tf.Tensor of shape=(len(sentences), 768)
  '''
  preprocessed_text = preprocessor(sentences)
  return encoder(preprocessed_text)['pooled_output']


#print(get_embeddings(["testing life forces in mind"]))


#Not necessary

def plot_similarity(features, labels):
    #Plot a similarity matrix of the embeddings.
    cos_sim = cosine_similarity(features)
    fig = plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.0)
    cbar_kws=dict(use_gridspec=False, location="left")
    g = sns.heatmap(
        cos_sim, xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, annot=True, cmap="Blues", 
        cbar_kws=cbar_kws)
    g.tick_params(labelright=True, labelleft=False)
    g.set_yticklabels(labels, rotation=0)
    g.set_title("Semantic Textual Similarity")
    plt.show()

reviews = ["cost of love is high",
           "price of love is extreme",
           "value of love is extraordinary"]

plot_similarity(get_embeddings(reviews), reviews)

#not necessary ends.

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
  return [np.argmax(pred) for pred in model.predict(reviews)]


#Defining a model as the preprocessor and encoder layers
#Follows a dropout and a dense layer with softmax activation function
i = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
x = preprocessor(i)
x = encoder(x)
x = tf.keras.layers.Dropout(0.2, name="dropout")(x['pooled_output'])
x = tf.keras.layers.Dense(num_classes, activation='softmax', name="output")(x)

#set the model's layers
model = tf.keras.Model(i, x)

#number of seasons the model should run for
n_epochs = 20

#List of metrics to be calculated by the model
METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
      balanced_recall,
      balanced_precision,
      balanced_f1_score
]

#if our model doesn't improve for 3 epochs -> patience = 3 we stop the training
#we restore the weights from the epoch where the validation loss showed the best value
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 3, restore_best_weights = True)

#compile the model with the parameters we selected
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = METRICS)

#fit the model using our train and validation data, callback as well.
model_fit = model.fit(X_train, y_train, epochs = n_epochs, validation_data = (X_valid, y_valid), callbacks = [earlystop_callback])

#evaluate the mode on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#create a classification report on the model performance based on the test set
y_pred = predict_class(X_test)
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
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert ground truth observations to one hot vectors
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

###########################################################
####### Still need to successfully save/load model ########
###########################################################
