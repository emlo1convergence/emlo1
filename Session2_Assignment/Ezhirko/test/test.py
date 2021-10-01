import requests
resp = requests.post("https://object-predict.herokuapp.com/predict",files={'file': open('car.png', 'rb')})
print(resp.text)