from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder

# ---------- Import data and convert categorical data to int ----------
df = pd.read_csv("../Data/cleaned_userdata.csv")

# Remove countries with less than 2 instances in the dataset (for stratified train/test splitting)
df = df.groupby('country').filter(lambda x: len(x) > 1)

print(len(df['country'].unique().tolist()))

# Ensure usernames and descriptions are in string format for parsing
df['description'] = df['description'].apply(lambda x: str(x))
df['username'] = df['username'].apply(lambda x: str(x))

# Integer encoding for countries
ienc = LabelEncoder().fit_transform(df['country'])
Y = ienc

corpus = df['username']

'''
# Tokenize & lemmatize
print("Lemmatizing...")
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df2 = pd.DataFrame()
corpus = df['description'].apply(lemmatize_text)

# Detokenize - this is probably silly but it works
def detokenize(text):
    return ' '.join(text)

corpus = corpus.apply(detokenize)
#print(corpus.head(5))
'''

gnb_bow_scores = list()
gnb_idf_scores = list()
lr_bow_scores = list()
lr_idf_scores = list()
svc_bow_scores = list()
svc_idf_scores = list()

for n in range(0, 99+1):

    # ---------- Bag-of-words ----------
    print("Vectorizing...")
    vectorizer = CountVectorizer()#strip_accents='ascii', stop_words='english', min_df=(1.0/corpus.count()))
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=n)

    #print("Fitting MultinomialNB...")
    gnb = MultinomialNB()
    gnb.fit(X_train, Y_train)
    score = gnb.score(X_test, Y_test)
    gnb_bow_scores.append(score)

    #print("Fitting Logistic Regression...")
    lr = LogisticRegression(multi_class='multinomial')
    lr.fit(X_train, Y_train)
    score = lr.score(X_test, Y_test)
    lr_bow_scores.append(score)

    #print("Fitting SVC")
    svc = SVC(kernel='linear', gamma='auto')
    svc.fit(X_train, Y_train)
    score = svc.score(X_test, Y_test)
    svc_bow_scores.append(score)

    # ---------- TF-IDF ----------

    vectorizer = TfidfVectorizer()#strip_accents='ascii', stop_words='english', min_df=(1.0/corpus.count()))
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=n)

    #print("Fitting MultinomialNB...")
    gnb = MultinomialNB()
    gnb.fit(X_train, Y_train)
    score = gnb.score(X_test, Y_test)
    gnb_idf_scores.append(score)

    #print("Fitting Logistic Regression...")
    lr = LogisticRegression(multi_class='multinomial')
    lr.fit(X_train, Y_train)
    score = lr.score(X_test, Y_test)
    lr_idf_scores.append(score)

    # SVC is slooooowwwww
    svc = SVC()
    svc.fit(X_train, Y_train)
    score = svc.score(X_test, Y_test)
    svc_idf_scores.append(score)

def Average(list):
    return (sum(list) / len(list))

print("Multinomal Naive Bayes BoW: " + str(Average(gnb_bow_scores)))
print("Support Vector Classification BoW: " + str(Average(svc_bow_scores)))
print("Logistic Regression BoW: " + str(Average(lr_bow_scores)))

print("Multinomal Naive Bayes TF-IDF: " + str(Average(gnb_idf_scores)))
print("Support Vector Classification TF-IDF: " + str(Average(svc_idf_scores)))
print("Logistic Regression TF-IDF: " + str(Average(lr_idf_scores)))