#############################################################################################################
# Use this to get a bunch of Tweets from the same user once their location has been determined              #
#############################################################################################################

import requests
import os
import json
import pandas as pd
import csv

os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADpGjQEAAAAAX%2FYQDcsFh9zgA0rLSwrNQim9b%2FM%3DJ0zkjfJbfjW8XeB720l8nLelsAtDe4bhib3gJJNOt0ihBBbUdw'
bearer_token = os.getenv('TOKEN')
headers = {"Authorization": "Bearer {}".format(bearer_token)}
max_results = 10

df = pd.read_csv('all_users.csv')
user_ids = df['id']
counter = 0

# The Twitter API returns the data in a random order, so don't just append all the csv files to each other or you get a mess
for id in user_ids:
    
    try:
        user_id = str(int(id))

        search_url = "https://api.twitter.com/2/users/" + str(user_id) + "/tweets"
        query_params = {#'expansions': 'author_id,geo.place_id',
                        'max_results': max_results,
                        #'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type',
                        #'tweet.fields': 'author_id,created_at,geo,id,lang,text',
                        'user.fields': 'description,id,location,name,username,verified'
                    }
        url = search_url, query_params # Maybe necessary?
        response = requests.request("GET", url[0], headers=headers, params=url[1])
        print("Endpoint response code: " + str(response.status_code))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        json_response = response.json()
        print(json.dumps(json_response, indent=4))
        
        df = pd.DataFrame(json_response['data'])
        df.to_csv('./UserData/data' + str(counter) + '.csv', mode='w', index=False, header=True)

        df = pd.DataFrame(json_response['includes'])
        df.to_csv('./UserData/includes' + str(counter) + '.csv', mode='w', index=False, header=False)

        counter = counter + 1

    except:
        print("Oops.")