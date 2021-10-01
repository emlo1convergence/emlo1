import requests

resp = requests.post("https://cifar10-predictor.herokuapp.com/predict",
                     files={"file": open('dog.jpg','rb')})

print(resp.text)