from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv("../Data/collated.csv")
Y = df['class']

corpus = df['text']
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