import requests

url = "http://127.0.0.1:8000/api/stt/transcribe/"
files = {'file': open('sample.mp3', 'rb')}
response = requests.post(url, files=files)
print(response.text)
