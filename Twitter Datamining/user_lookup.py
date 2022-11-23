#############################################################################################################
# Use this to get a bunch of Tweets from the same user once their location has been determined              #
#############################################################################################################

import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time

os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADpGjQEAAAAA3Do%2Ft8nLNUMYrs%2FKlGFs00ioHps%3D6zs5KsUh5BSjclj1kre97JRq3yMyHsVYC8OsF88a4naIuwFfPE'
bearer_token = os.getenv('TOKEN')
headers = {"Authorization": "Bearer {}".format(bearer_token)}
max_results = 100

df = pd.read_csv('users_compact.csv')
user_ids = df['id']
counter = 0

# The Twitter API returns the data in a random order, so don't just append all the csv files to each other or you get a mess
for id in user_ids:
    
    user_id = id

    search_url = "https://api.twitter.com/2/users/" + str(user_id) + "/tweets"
    query_params = {'expansions': 'author_id,geo.place_id',
                    'max_results': max_results,
                    'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type',
                    'tweet.fields': 'author_id,created_at,geo,id,lang,text',
                    'user.fields': 'id,location,name,username,verified'
                }
    url = search_url, query_params # Maybe necessary?
    response = requests.request("GET", url[0], headers=headers, params=url[1])
    print("Endpoint response code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    json_response = response.json()
    print(json.dumps(json_response, indent=4))
    df = pd.DataFrame(json_response['data'])

    df.to_csv('./Tweets/' + str(counter) + '.csv', mode='w', index=False, header=True)
    counter = counter + 1