from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.utils import to_categorical
import pandas as pd
import nltk

# Import data and convert categorical data to int
df = pd.read_csv("data.csv")
countries = df['country']
cat = pd.Categorical(df['country'])
country_code = pd.Series(cat.codes)

# One Hot Encoding of countries
#Y = to_categorical(country_code, num_classes=78)
Y = country_code

# Tokenize & lemmatize
print("Lemmatizing...")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df2 = pd.DataFrame()
corpus = df['text'].apply(lemmatize_text)

# Detokenize - this is probably silly but it works
def detokenize(text):
    return ' '.join(text)

corpus = corpus.apply(detokenize)
print(corpus.head(5))

# Bag-of-words
print("Vectorizing...")
vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', min_df=(100.0/corpus.count()))
X = vectorizer.fit_transform(corpus)
print(X.shape)

# Divvy up the Tweet data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print("Fitting MultinomialNB...")
gnb = MultinomialNB()
gnb.fit(X_train, Y_train)
print("MultinomialNB Bag-of-words score: " + str(gnb.score(X_test, Y_test)))

print("Fitting SVC")
svc = SVC(kernel='linear', gamma='auto')
svc.fit(X_train, Y_train)
print("SVC Bag-of-words score: " + str(svc.score(X_test, Y_test)))

print("Fitting Logistic Regression...")
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, Y_train)
print("LR Bag-of-words score: " + str(lr.score(X_test, Y_test)))

"""
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus) # Bag-of-words vectorization
#print(vectorizer.get_feature_names_out())
#print(X.toarray())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb = MultinomialNB()
gnb.fit(X_train.toarray(), Y_train)
print("MultinomialNB Bag-of-words score: " + str(gnb.score(X_test.toarray(), Y_test)))

svc = SVC()
svc.fit(X_train, Y_train)
print("SVC Bag-of-words score: " + str(svc.score(X_test, Y_test)))

lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, Y_train)
print("LR Bag-of-words score: " + str(lr.score(X_test, Y_test)))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
#print(X.toarray())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb = MultinomialNB()
gnb.fit(X_train.toarray(), Y_train)
print("MultinomialNB TF-IDF score: " + str(gnb.score(X_test.toarray(), Y_test)))

svc = SVC()
svc.fit(X_train, Y_train)
print("SVC TF-IDF score: " + str(svc.score(X_test, Y_test)))

lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, Y_train)
print("LR TF-IDF: " + str(lr.score(X_test, Y_test)))
"""