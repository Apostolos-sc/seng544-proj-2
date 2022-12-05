import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

def preprocessing(text_column):

    # Remove "RT" from begining of sentence - do first cause RT is capitalized.
    df[text_column] = df[text_column].map(lambda name: re.sub('^(RT)', ' ', name))
    
    # Remove links
    df[text_column] = df[text_column].map(lambda name: re.sub(r'http\S+', ' ', name))
    
    # Remove mentions
    df[text_column] = df[text_column].map(lambda name: re.sub("@([a-zA-Z0-9_]{1,50})", '', name))

    # Remove repeated instances of characters
    #repeat_pattern = re.compile(r'(\w)\1*') #compile the pattern we are looking for
    #match_substitution = r'\1' #substituion pattern
    #df[text_column] = df[text_column].map(lambda name: re.sub(repeat_pattern, match_substitution, name))

    # Remove of digits with regex - we do this here because it is possible to have numbers in tags and urls replace with space.
    df[text_column] = df[text_column].map(lambda name: re.sub(r'[0-9]', ' ', name))
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
    df[text_column] = df[text_column].map(lambda name: re.sub(EMOJI_PATTERN, ' ', name))

    # Do this after removing mentions -> don't # here. ->replace with space.
    df[text_column] = df[text_column].map(lambda name: name.lower())

    # Remove most punctuation
    special_pattern = re.compile('[!\.\^\$\|\?\*\+\=\(\)\{\}\@\=\/\<\>\,\~\`\-\%\&\:\;\[\]"“”…]')
    df[text_column] = df[text_column].map(lambda name: re.sub(special_pattern, ' ', name))

    # Remove a hashtag if it has no significance, ie, not part of a #word
    df[text_column] = df[text_column].map(lambda name: re.sub('(#[^(a-zA-Z0-9)])', ' ', name))

    # Removing doublicate spaces and all white spaces like \t, \n or \r
    df[text_column] = df[text_column].map(lambda name: " ".join(name.split()))

    # Now remove stop words
    df[text_column] = df[text_column].map(lambda name: ' '.join([word for word in name.split() if word not in stopwords_dict]))

    # After removing stop words we can clean up more
    df[text_column] = df[text_column].map(lambda name: re.sub('[\']', ' ', name))

    # Final white space clean up
    df[text_column] = df[text_column].map(lambda name: " ".join(name.split(' ')))

    # Still need to check for strings that contain whitespaces only and remove them
    df[text_column] = df[text_column].map(lambda text: np.nan if len(text) == 0 else text)
    df.dropna(axis=0, inplace=True)

    # Export results to csv
    df.to_csv("cleaned_tweets.csv")

df = pd.read_csv("tweets.csv")
text = 'text'
df[text] = df[text].apply(lambda x: str(x))
preprocessing(text)

#text = 'username'
#df[text] = df[text].apply(lambda x: str(x))
#preprocessing(text)