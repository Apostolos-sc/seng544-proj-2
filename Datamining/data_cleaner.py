############################################################
# Reads multiple csv files and collates them into one file #
############################################################

import pandas as pd
import numpy as np

df = pd.DataFrame()

for n in range(0, 758+1):
    file_path = './Tweets2/' + str(n) + '.csv'
    tweets = pd.read_csv(file_path)
    temp = pd.DataFrame()
    temp['user_id'] = tweets['author_id']
    temp['created_at'] = tweets['created_at']
    temp['tweet_id'] = tweets['id']
    temp['text'] = tweets['text']

    df = pd.concat([df, temp], axis=0)

user_data = pd.read_csv('users_compact2.csv', index_col=False)

df = df.merge(user_data, left_on='user_id', right_on='user_id', how='left')

#print(df.head(10))
df.to_csv('out2.csv')