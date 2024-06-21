import requests

url = 'http://127.0.0.1:8887/detect_from_bytes'
files = [('files', open('../../../../data/examples/oneline_images/example1.jpeg', 'rb')),
         ('files', open('../../../../data/examples/oneline_images/example2.jpeg', 'rb'))]
resp = requests.post(url=url, files=files)
resp.raise_for_status()
print(resp.json())
