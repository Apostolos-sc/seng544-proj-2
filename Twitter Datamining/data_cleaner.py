import pandas as pd
import numpy as np

df = pd.DataFrame()

user_data = 

for n in range(0, 605+1):
    file_path = './Tweets/' + str(n) + '.csv'
    tweets = pd.read_csv(file_path)
    temp = pd.DataFrame()
    temp['user_id'] = tweets['author_id']
    temp['text'] = tweets['text']
    temp['created_at'] = tweets['created_at']

    df = pd.concat([df, temp], axis=0)

print(df.head())
print(df.count())

df.to_csv('out.csv')