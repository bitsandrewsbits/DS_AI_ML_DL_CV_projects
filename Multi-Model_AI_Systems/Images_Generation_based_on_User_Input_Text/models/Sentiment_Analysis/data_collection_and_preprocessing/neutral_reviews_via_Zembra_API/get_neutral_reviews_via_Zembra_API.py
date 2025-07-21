import json
import requests
import openapi_client
from openapi_client.rest import ApiException
import pandas as pd

sandbox_token = "your_zembra_sandbox_token"
network_name = 'google'
slug = "ChoIzsbB-eTP8MqzARoNL2cvMTFiNXZfMWdoZhABs"

configuration = openapi_client.Configuration(
  host = "https://sandbox.zembra.io",
  access_token = f"Bearer {sandbox_token}"
)

reviews_max_amount = 2

def get_neutral_reviews_via_ReviewsAPI(reviews_max_amount = 2) -> pd.DataFrame:
  neutral_reviews = {"text": [], "label": []}
  with openapi_client.ApiClient(configuration) as api_client:
    api_instance = openapi_client.ReviewsApi(api_client)
    network = openapi_client.ReviewNetwork(network_name)
    try:
      api_response = api_instance.get_review_job(
        network = network, slug = slug, fields = ['text', 'rating'], 
        has_replies = False, 
        min_rating = 3, max_rating = 3, language = 'en',
        include_deleted = True, limit = reviews_max_amount
      )
      reviews_objects = api_response.data.reviews
      for review_obj in reviews_objects:
        neutral_reviews["text"].append(review_obj.text)
        neutral_reviews["label"].append(review_obj.rating)
    except ApiException as e:
      print(e)
  print(pd.DataFrame(neutral_reviews))
  return 0

if __name__ == "__main__":
  get_neutral_reviews_via_ReviewsAPI(reviews_max_amount)
