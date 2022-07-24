import requests 

file = '0_0.jpg'

# https://your-heroku-app-name.herokuapp.com/predict
resp = requests.post("http://127.0.0.1:5000/eyedisease", files={'file': open(file, 'rb')})

print(resp.text)