##########################
### Text Preprocessing ###
##########################
#For easy reference
import pandas as pd
import re

df = pd.read_csv('tweets.csv')

#remove rt from begining of sentence - do first cause RT is capitalized.
df["text"] = df["text"].map(lambda name: re.sub('^(RT)', ' ', name))

#removing links
df["text"] = df["text"].map(lambda name: re.sub(r'http\S+', ' ', name))

df.to_csv("chatbot_tweets.csv", encoding='utf-8')