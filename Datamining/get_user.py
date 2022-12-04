#############################################################################################################
# Use this to get a bunch of Twitter users, then check if they have an acutal location in their description #
#############################################################################################################

# For sending GET requests from the API
import requests

# For saving access tokens and for file management when creating and adding to the dataset
import os

# For dealing with json response we receive from the API
import json

# For displaying the data after
import pandas as pd

# For saving the response data in CSV format
import csv

# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata

# To add wait time between requests
import time

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

# Prepare the API request
def create_url(keyword, start_date, end_date, max_results):
    
    search_url = "https://api.twitter.com/2/tweets/search/recent"

    # See: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent for a list
    query_params = {'query': keyword,
                    'max_results': max_results,
                    'expansions': 'author_id',
                    'tweet.fields': 'id,text,author_id,geo,created_at,lang',
                    'user.fields': 'id,name,username,created_at,description,location,protected,public_metrics,verified',
                    'place.fields': 'full_name,id,contained_within,country,country_code,geo,name,place_type',
                    'next_token': {}
                    }
    return (search_url, query_params)

# Send the request
# Not sure how the next_token bit works yet
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   # params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

# API Key, don't share outside of the project group :P
os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAADpGjQEAAAAA3Do%2Ft8nLNUMYrs%2FKlGFs00ioHps%3D6zs5KsUh5BSjclj1kre97JRq3yMyHsVYC8OsF88a4naIuwFfPE'

#Inputs for the request
bearer_token = auth()
headers = create_headers(bearer_token)
start_time = "2021-03-01T00:00:00.000Z"
end_time = "2021-03-31T00:00:00.000Z"
max_results = 100

# !!! Change these each search !!!
keyword = "lang:en location"
file_number = 30

# Create the query
url = create_url(keyword, start_time, end_time, max_results)
json_response = connect_to_endpoint(url[0], headers, url[1])
print(json.dumps(json_response, indent=4))

# Export data to csv files
df = pd.DataFrame(json_response['data'])
df.to_csv('./Users/data' + str(file_number) + '.csv', mode='w', index=False, header=False)
df = pd.DataFrame(json_response['includes'])
df.to_csv('./Users/includes' + str(file_number) + '.csv', mode='w', index=False, header=False)

# Go to: https://twitter.com/anyuser/status/[tweet id] to find a specific tweet by id

# Tweets by user id:
# https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/api-reference/get-users-id-tweets
# See other .py file in this dir for implementation