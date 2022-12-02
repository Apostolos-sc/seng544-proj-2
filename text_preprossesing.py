##########################
### Text Preprocessing ###
##########################
#For easy reference
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import re
import numpy as np
#nltk.download('stopwords')
# for Colab users: !pip install tensorflow_text
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

df = pd.read_csv('data.csv')
#df.head()
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
