import json
import requests

sandbox_token = "your_test_Zembra_token"
network_name = 'google'
slug = "ChoIzsbB-eTP8MqzARoNL2cvMTFiNXZfMWdoZhABs"
host = "https://sandbox.zembra.io"

payload = {
    "monitoring": 'full'
}
headers = {
  'Authorization': f'Bearer {sandbox_token}'
}
reviews_amount = 2

url = f"{host}/reviews?network={network_name}&slug={slug}&limit={reviews_amount}"
response = requests.request("GET", url, headers=headers, data=payload)
response_data = response.text
reviews_json = json.loads(response_data)
print(reviews_json["data"]["reviews"])
