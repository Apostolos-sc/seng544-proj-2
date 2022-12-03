from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# ---------- Import data and convert categorical data to int ----------
df = pd.read_csv("../Data/cleaned_userdata.csv")
countries = df['country']
cat = pd.Categorical(df['country'])
country_code = pd.Series(cat.codes)

print(df['country'].tail(10))
print(country_code.tail(10))

df['description'] = df['description'].apply(lambda x: str(x))
print(df['description'])

# Integer encoding
ienc = LabelEncoder().fit_transform(df['country'])
Y = ienc
#print(ienc)

# One hot encoding
ohe = OneHotEncoder(sparse=False)
ienc = ienc.reshape(len(ienc), 1)
#Y = ohe.fit_transform(ienc)
#print(Y)

# Tokenize & lemmatize
print("Lemmatizing...")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df2 = pd.DataFrame()
corpus = df['username'].apply(lemmatize_text)

# Detokenize - this is probably silly but it works
def detokenize(text):
    return ' '.join(text)

corpus = corpus.apply(detokenize)
print(corpus.head(5))

# ---------- Bag-of-words ----------
print("Vectorizing...")
vectorizer = CountVectorizer()#strip_accents='ascii', stop_words='english', min_df=(1.0/corpus.count()))
X = vectorizer.fit_transform(corpus)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

print("Fitting MultinomialNB...")
gnb = MultinomialNB()
gnb.fit(X_train, Y_train)
#print(gnb.class_count_)
print("MultinomialNB Bag-of-words score: " + str(gnb.score(X_test, Y_test)))

print("Fitting Logistic Regression...")
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, Y_train)
print("LR Bag-of-words score: " + str(lr.score(X_test, Y_test)))

# SVC is slooooowwwww
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.1, random_state=0)
#X_train2, X_test, Y_train2, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

print("Fitting SVC")
svc = SVC(kernel='linear', gamma='auto')
svc.fit(X_train, Y_train)
print("SVC Bag-of-words score: " + str(svc.score(X_test, Y_test)))

# ---------- TF-IDF ----------

vectorizer = TfidfVectorizer()#strip_accents='ascii', stop_words='english', min_df=(1.0/corpus.count()))
X = vectorizer.fit_transform(corpus)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)

print("Fitting MultinomialNB...")
gnb = MultinomialNB()
gnb.fit(X_train, Y_train)
print("MultinomialNB TF-IDF score: " + str(gnb.score(X_test, Y_test)))
print(gnb.predict(X_test))

print("Fitting Logistic Regression...")
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, Y_train)
print("LR TF-IDF: " + str(lr.score(X_test, Y_test)))

# SVC is slooooowwwww
svc = SVC()
svc.fit(X_train, Y_train)
print("SVC TF-IDF score: " + str(svc.score(X_test, Y_test)))