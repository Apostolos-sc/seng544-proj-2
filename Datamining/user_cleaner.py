####################################################################################
# Collates usernids, userames, user descriptions, and locations into one dataframe #
####################################################################################

import pandas as pd
import numpy as np
import regex as re

#user_locations = pd.DataFrame()
user_ids = list()
user_locs = list()
user_names = list()
user_descs = list()

#for n in range(0, 30+1):
#file_path = './Users/includes' + str(n) + '.csv'
file_path = 'includes.csv'
users = pd.read_csv(file_path)

for u in range(1, len(users)):
    
    list = users.iloc[u][0].split(' \'')
    i = 0
    temp = pd.DataFrame()

    id_found = str()
    loc_found = str()
    name_found = str()
    desc_found = str()

    for s in list:
        user_id = bool(re.match('id', s))
        location = bool(re.match('location', s))
        user_name = bool(re.match('username',s))
        user_desc = bool(re.match('description', s))
    
        if (user_id):
            #print(list[i+1])
            if (len(list) > i+1):
                id_found = list[i+1]
                print(list[i+1])
            else:
                id_found = 0
        if (location): 
            #print(list[i+1])
            if (len(list) > i+1):
                loc_found = list[i+1]
        if (user_name):
            if (len(list) > i+1):
                name_found = list[i+1]
        if (user_desc):
            if (len(list) > i+1):
                desc_found = list[i+1]
        
        i += 1

    user_ids.append(id_found)
    user_names.append(name_found)
    user_descs.append(desc_found)

    if not loc_found:
        user_locs.append("")
    else:
        user_locs.append(loc_found)

print(len(user_ids))
print(len(user_locs))
print(len(user_names))
print(len(user_descs))

user_locations = pd.DataFrame()
user_locations['user_id'] = user_ids
user_locations['location'] = user_locs
user_locations['username'] = user_names
user_locations['description'] = user_descs

user_locations.to_csv('user_data2.csv', mode='w')