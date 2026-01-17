import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------------------
# Model initialization (ONCE)
# -------------------------------

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval()

# -------------------------------
# Path to stored face embeddings
# -------------------------------

FACES_DIR = "data/registered_faces"


# -------------------------------
# Extract face embedding
# -------------------------------

def extract_embedding(image_path):
    """
    Input: image path
    Output: 512-d face embedding (numpy array)
    """

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Detect and crop face
    face = mtcnn(img)

    if face is None:
        return None

    # Add batch dimension
    face = face.unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        embedding = resnet(face)

    return embedding.numpy()[0]


# -------------------------------
# Register a new user face
# -------------------------------

def register_face(user_id, image_path):
    """
    Registers a user's face by saving embedding
    """

    embedding = extract_embedding(image_path)

    if embedding is None:
        raise ValueError("No face detected in image")

    os.makedirs(FACES_DIR, exist_ok=True)

    save_path = os.path.join(FACES_DIR, f"{user_id}.npy")
    np.save(save_path, embedding)

    print(f"[INFO] Face registered for user: {user_id}")


# -------------------------------
# Recognize a face
# -------------------------------

def recognize_face(image_path, threshold=0.9):
    """
    Matches face against registered users
    Returns (user_id, distance)
    """

    embedding = extract_embedding(image_path)

    if embedding is None:
        return None, None

    best_match = None
    best_distance = float("inf")

    if not os.path.exists(FACES_DIR):
        return None, None

    for file in os.listdir(FACES_DIR):
        if not file.endswith(".npy"):
            continue

        user_id = file.replace(".npy", "")
        saved_embedding = np.load(
            os.path.join(FACES_DIR, file)
        )

        distance = np.linalg.norm(
            embedding - saved_embedding
        )

        if distance < best_distance:
            best_distance = distance
            best_match = user_id

    if best_distance < threshold:
        return best_match, best_distance
    else:
        return None, best_distance
