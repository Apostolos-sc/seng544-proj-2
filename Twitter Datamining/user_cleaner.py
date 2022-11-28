import pandas as pd
import numpy as np
import regex as re

#user_locations = pd.DataFrame()
user_ids = list()
user_locs = list()

for n in range(0, 30+1):
    file_path = './Users/includes' + str(n) + '.csv'
    users = pd.read_csv(file_path)

    for u in range(1, len(users)):
        
        list = users.iloc[u][0].split(' \'')
        i = 0
        temp = pd.DataFrame()

        id_found = str()
        loc_found = str()

        for s in list:
            user_id = bool(re.match('id', s))
            location = bool(re.match('location', s))
        
            if (user_id):
                #print(list[i+1])
                if (len(list) > i+1):
                    id_found = [list[i+1]]
                else:
                    id_found = 0
            if (location): 
                #print(list[i+1])
                if (len(list) > i+1):
                    loc_found = list[i+1]
            
            i += 1

        user_ids.append(id_found)

        if not loc_found:
            user_locs.append("")
        else:
            user_locs.append(loc_found)

print(len(user_ids))
print(len(user_locs))

user_locations = pd.DataFrame()
user_locations['user_id'] = user_ids
user_locations['location'] = user_locs

user_locations.to_csv('user_locations.csv', mode='w')

#for n in range(0, 30+1):
#    file_path = './Users/includes' + str(n) + '.csv'
#    tweets = pd.read_csv(file_path)
#    temp = pd.DataFrame()
#    temp['user_id'] = tweets['author_id']
#    temp['created_at'] = tweets['created_at']
#    temp['tweet_id'] = tweets['id']
#    temp['text'] = tweets['text']

#    df = pd.concat([df, temp], axis=0)

#user_data = pd.read_csv('users_compact.csv', index_col=False)

#df = df.merge(user_data, left_on='user_id', right_on='id', how='left')

#print(df.head(10))
#df.to_csv('out.csv')