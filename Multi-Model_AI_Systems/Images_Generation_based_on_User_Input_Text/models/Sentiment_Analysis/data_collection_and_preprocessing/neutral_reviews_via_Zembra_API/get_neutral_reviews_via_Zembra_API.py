import json
import requests
import openapi_client
from openapi_client.rest import ApiException
import pandas as pd

API_token = "your_token"
network_name = "appstore"
slug = "1523383806"

configuration = openapi_client.Configuration(
  host = "https://api.zembra.io",
  access_token = f"Bearer {API_token}"
)

reviews_max_amount = 10

def get_neutral_reviews_via_ReviewsAPI(reviews_max_amount = 2) -> pd.DataFrame:
  neutral_reviews = {"text": [], "label": []}
  with openapi_client.ApiClient(configuration) as api_client:
    api_instance = openapi_client.ReviewsApi(api_client)
    network = openapi_client.ReviewNetwork(network_name)
    try:
      api_response = api_instance.get_review_job(
        network = network, slug = slug, fields = ['text', 'rating'],
        min_rating = 2.5, max_rating = 4,
        limit = reviews_max_amount
      )
      reviews_objects = api_response.data.reviews
      for review_obj in reviews_objects:
        neutral_reviews["text"].append(review_obj.text)
        neutral_reviews["label"].append(review_obj.rating)
    except ApiException as e:
      print(e)
  print(pd.DataFrame(neutral_reviews))
  return pd.DataFrame(neutral_reviews)

def save_neutral_reviews_as_JSON_file(dataframe: pd.DataFrame):
  dataframe.to_json("neutral_reviews.json", orient = 'records', lines = True)

if __name__ == "__main__":
  neutral_reviews_df = get_neutral_reviews_via_ReviewsAPI(reviews_max_amount)
  save_neutral_reviews_as_JSON_file(neutral_reviews_df)