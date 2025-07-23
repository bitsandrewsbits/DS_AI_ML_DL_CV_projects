import json
import openapi_client
from openapi_client.models.create_review_job_request import CreateReviewJobRequest
from openapi_client.rest import ApiException
import pandas as pd

API_token = "your_API_token"
network_name = "google"
slug = "ChIJB0ZMDJ903IARTmMwT37ClbM"

configuration = openapi_client.Configuration(
  host = "https://api.zembra.io",
  access_token = f"Bearer {API_token}"
)

def get_neutral_reviews_via_ReviewsAPI(max_reviews_from_API = 1000) -> pd.DataFrame:
  neutral_reviews = {"text": [], "label": []}
  with openapi_client.ApiClient(configuration) as api_client:
    api_instance = openapi_client.ReviewsApi(api_client)
    network = openapi_client.ReviewNetwork(network_name)
    job_request_json = json.dumps({
      "network": network_name,
      "slug": slug,
      "monitoring": 'full',
      "min_rating": 3,
      "max_rating": 3
    })
    # review_job_request = openapi_client.CreateReviewJobRequest.from_json(job_request_json)   # uncomment if start first time
    job_id = "3a753e21-b383-408f-a485-92749e1db1d4"
    uid = "8a8be640-16d6-40dc-885f-71c207b2e75f"
    fields=["text", "rating"]
    try:
      # reviews_job = api_instance.create_review_job(review_job_request)    # uncomment if start first time
      # print(reviews_job)        # uncomment if start first time
      api_response = api_instance.get_review_job(
        network = network, 
        slug = slug,
        job_id = job_id, 
        uid = uid, 
        fields = fields,
        min_rating = 3.0,
        max_rating = 3.0,
        limit = max_reviews_from_API,
      )
      reviews_objects = api_response.data.reviews
      for review_obj in reviews_objects:
        if review_obj.text != "" and review_obj.text != None:
          neutral_reviews["text"].append(review_obj.text)
          neutral_reviews["label"].append(review_obj.rating)
    except ApiException as e:
      print(e)
  print(pd.DataFrame(neutral_reviews))
  return pd.DataFrame(neutral_reviews)

def save_neutral_reviews_as_JSON_file(dataframe: pd.DataFrame):
  dataframe.to_json("neutral_reviews.json", orient = 'records', lines = True)

if __name__ == "__main__":
  neutral_reviews_df = get_neutral_reviews_via_ReviewsAPI()
  save_neutral_reviews_as_JSON_file(neutral_reviews_df)