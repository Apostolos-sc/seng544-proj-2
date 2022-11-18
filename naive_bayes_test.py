from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("collated.csv")
Y = df['class']

corpus = df['text']
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus) # Bag-of-words vectorization
#print(vectorizer.get_feature_names_out())
#print(X.toarray())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train.toarray(), Y_train)
print("Bag-of-words score: " + str(gnb.score(X_test.toarray(), Y_test)))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
#print(X.toarray())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb = GaussianNB()
gnb.fit(X_train.toarray(), Y_train)
print("TF-IDF score: " + str(gnb.score(X_test.toarray(), Y_test)))