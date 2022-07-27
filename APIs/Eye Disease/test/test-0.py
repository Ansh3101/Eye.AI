import requests 

file = '/Users/anshumantekriwal/Desktop/Uber Global Hackathon/API/Eye Disease/test/0_0.jpg'

resp = requests.post("https://eyeai-cornealulcers.herokuapp.com/", files={'file': open(file, 'rb')})

print(resp.text)