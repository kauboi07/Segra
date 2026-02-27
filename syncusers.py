import os
import requests
from face_recognition.face_utils import register_face

BASE_URL = "http://127.0.0.1:8000/api/v1"
USERS_LIST_URL = f"{BASE_URL}/users/"
INCOMING_DIR = "data/incoming"
REGISTERED_DIR = "data/registered_faces"

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(REGISTERED_DIR, exist_ok=True)

response = requests.get(USERS_LIST_URL)

if response.status_code != 200:
    print(f"[ERROR] Failed to fetch users list: {response.status_code}")
    exit(1)

users = response.json()
for user in users:
    username = user["username"]
    has_photo = user["has_photo"]
    uid=user["id"]

    if not has_photo:
        print(f"[SKIP] {username} has no profile photo")
        continue

    embedding_path = os.path.join(REGISTERED_DIR, f"{username}.npy")
    if os.path.exists(embedding_path):
        print(f"[SKIP] {username} already registered")
        continue

    face_url = f"{BASE_URL}/users/{uid}/face/"
    face_response = requests.get(face_url)

    if face_response.status_code != 200:
        print(f"[FAIL] Could not fetch face for {username}")
        continue

    image_path = os.path.join(INCOMING_DIR, f"{username}.jpg")
    with open(image_path, "wb") as f:
        f.write(face_response.content)

    try:
        register_face(username, image_path)
        print(f"[INFO] Registered {username}")
    except Exception as e:
        print(f"[FAIL] Registration failed for {username}: {e}")
