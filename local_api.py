import os
import json
import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
URL = 'http://127.0.0.1:8000'

r = requests.get(URL)

print(r.status_code)
print(r.text)



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

post_url = URL+'/data/'
r = requests.post(post_url, json = data)

print(r.status_code)
print(r.text)
