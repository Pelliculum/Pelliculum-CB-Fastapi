import requests

response = requests.post(
    "http://localhost:8000/recommend/",
    json={"title": "Fast X"}
)
print(response.json())