import requests

url = "http://127.0.0.1:5000/process"

image_path = r"C:/Users/123co/OneDrive/Pictures/Camera Roll/WIN_20260117_14_30_48_Pro.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

#print(response.json())
print("Status code:", response.status_code)
print("Raw response:")
print(response.text)
