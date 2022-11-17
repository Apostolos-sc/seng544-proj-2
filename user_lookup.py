import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time

os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADpGjQEAAAAAziwCga2vbnOWBQSePjtuFvx0bL0%3DVNRKp7h5BE3gRL1tSTimWaWbNULvyjHaZVvgKIoOQVHyKu4cgE'

bearer_token = os.getenv('TOKEN')
headers = {"Authorization": "Bearer {}".format(bearer_token)}
max_results = 100
user_id = 25029937
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
df.to_csv('tweets.csv')