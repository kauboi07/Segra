import os
from flask import Blueprint, request, jsonify
from face_recognition.face_utils import recognize_face

process_api = Blueprint("process_api", __name__)

INCOMING_DIR = "data/incoming"

@process_api.route("/process", methods=["POST"])
def process_image():
    # 1. Check if image is sent
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    # 2. Save image locally
    os.makedirs(INCOMING_DIR, exist_ok=True)
    image_path = os.path.join(INCOMING_DIR, "input.jpg")
    image.save(image_path)

    # 3. Run face recognition
    user_id, distance = recognize_face(image_path)

    if user_id is None:
        return jsonify({
            "user_id": None,
            "confidence": None,
            "message": "Unknown user"
        })

    # 4. Return result
    confidence = round(float(1 - distance), 3)

    return jsonify({
        "user_id": user_id,
        "confidence": confidence
    })
