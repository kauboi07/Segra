from flask import Flask, jsonify, send_file
import requests
import os
import time
from face_recognition.face_utils import recognize_face
from wasteclassifier import classify_waste
from flask import render_template_string

app = Flask(__name__)

ESP_IP = "http://10.211.145.165"
INCOMING_DIR = "data/incoming"
DJANGO_API_BASE = os.getenv("DJANGO_API_BASE", "http://localhost:8000/api")

os.makedirs(INCOMING_DIR, exist_ok=True)


#capture image from esp32 cam
def capture_image(filename):
    try:
        response = requests.get(f"{ESP_IP}/capture", timeout=10)

        if response.status_code != 200:
            return None

        image_path = os.path.join(INCOMING_DIR, filename)

        with open(image_path, "wb") as f:
            f.write(response.content)

        return image_path

    except Exception:
        return None


#get uid
def get_user_id_by_username(username):
    try:
        response = requests.get(f"{DJANGO_API_BASE}/v1/users/", timeout=10)

        if response.status_code == 200:
            users = response.json()
            for user in users:
                if user.get("username") == username:
                    return user.get("id")

        return None

    except Exception:
        return None


#give points
def award_points(username, waste_type):
    user_id = get_user_id_by_username(username)

    if user_id is None:
        print("User not found in Django:", username)
        return None

    payload = {
        "user_id": user_id,
        "waste_type": waste_type,
        "weight": 1.0
    }

    try:
        response = requests.post(
            f"{DJANGO_API_BASE}/v1/iot/dispose/",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        print("error:", response.text)
        return None

    except Exception:
        return None

WASTE_TYPE_MAP = {
    "organic": "bio",
    "plastic": "plastic",
    "metal": "metal",
    "ewaste": "ewaste"
}


#flow
@app.route("/start", methods=["GET"])
def start_process():
    try:
        # face
        print("Capturing face...")
        face_image = capture_image("face.jpg")

        if face_image is None:
            return jsonify({"error": "Failed to capture face image"}), 500

        user, distance = recognize_face(face_image)

        if user is None:
            return jsonify({
                "status": "unknown",
                "message": "User not recognized"
            })

        confidence = float(round(1 - float(distance), 3))

        print(f"User recognized: {user}")
        print("Waiting 5 seconds for waste...")

        time.sleep(5)

        # object
        print("Capturing waste...")
        waste_image = capture_image("waste.jpg")

        if waste_image is None:
            return jsonify({"error": "Failed to capture waste image"}), 500

        waste_type = classify_waste(waste_image)
        print(f"Waste detected: {waste_type}")

        django_type = WASTE_TYPE_MAP.get(waste_type, waste_type)

        #points
        points_response = award_points(user, django_type)

        return jsonify({
            "status": "success",
            "user": user,
            "confidence": confidence,
            "waste_type": waste_type,
            "points": points_response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


'''
@app.route("/last_image/<image_type>", methods=["GET"])
def last_image(image_type):
    if image_type not in ["face", "waste"]:
        return jsonify({"error": "Invalid image type"}), 400

    image_path = os.path.join(INCOMING_DIR, f"{image_type}.jpg")

    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/jpeg")
    else:
        return jsonify({"error": "Image not found"}), 404
'''

#preview page(testing not in production) - in prod we will replace this webapge
#with a physical button so when user clicks that the process will start it is not implemented yet.
@app.route("/preview")
def preview():
    html = f"""
    <html>
    <head>
        <title>Smart Bin Live Preview</title>
        <style>
            body {{
                text-align: center;
                font-family: Arial;
                background: #111;
                color: white;
            }}
            img {{
                width: 640px;
                border: 4px solid #00ffcc;
                border-radius: 10px;
            }}
            button {{
                margin-top: 20px;
                padding: 10px 20px;
                font-size: 18px;
            }}
        </style>
    </head>
    <body>
        <h2>Smart Bin Camera Preview</h2>
        <img src="{ESP_IP}/stream">
        <br>
        <button onclick="window.location.href='/start'">
            Start Smart Bin Process
        </button>
    </body>
    </html>
    """
    return render_template_string(html)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)