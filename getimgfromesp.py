from flask import Flask, jsonify, render_template_string
import requests
import os
import time
from face_recognition.face_utils import recognize_face
from wasteclassifier import classify_waste

app = Flask(__name__)

# ==========================
# IP CONFIG
# ==========================
CAMERA_IP = "http://10.211.145.165"   # ESP32-CAM
SERVO_IP  = "http://10.211.145.254"   # Servo ESP

DJANGO_API_BASE = os.getenv("DJANGO_API_BASE", "http://localhost:8000/api")

INCOMING_DIR = "data/incoming"
os.makedirs(INCOMING_DIR, exist_ok=True)

# ==========================
# BIN MAPPING
# ==========================
BIN_ENDPOINT_MAP = {
    "organic": "s1",
    "plastic": "s2",
    "metal": "s3",
    "ewaste": "s4"
}

WASTE_TYPE_MAP = {
    "organic": "bio",
    "plastic": "plastic",
    "metal": "metal",
    "ewaste": "ewaste"
}

# ==========================
# CAMERA CAPTURE
# ==========================
def capture_image(filename):
    try:
        response = requests.get(f"{CAMERA_IP}/capture", timeout=10)
        if response.status_code != 200:
            return None

        path = os.path.join(INCOMING_DIR, filename)
        with open(path, "wb") as f:
            f.write(response.content)

        return path
    except Exception as e:
        print("Camera error:", e)
        return None

# ==========================
# DJANGO HELPERS
# ==========================
def get_user_id_by_username(username):
    try:
        response = requests.get(f"{DJANGO_API_BASE}/v1/users/", timeout=10)
        if response.status_code == 200:
            users = response.json()
            for user in users:
                if user.get("username") == username:
                    return user.get("id")
        return None
    except Exception as e:
        print("Django fetch error:", e)
        return None


def award_points(username, waste_type):
    user_id = get_user_id_by_username(username)
    if user_id is None:
        print("User not found:", username)
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

        print("Django error:", response.text)
        return None
    except Exception as e:
        print("Django award error:", e)
        return None

# ==========================
# SERVO CONTROL
# ==========================
def open_bin(waste_type):
    endpoint = BIN_ENDPOINT_MAP.get(waste_type)

    if not endpoint:
        print("No bin mapping for:", waste_type)
        return

    try:
        # Open servo
        requests.get(f"{SERVO_IP}/{endpoint}/open", timeout=5)

        # Keep open 2 seconds
        time.sleep(2)

        # Close servo
        requests.get(f"{SERVO_IP}/{endpoint}/close", timeout=5)

        print(f"Opened bin for {waste_type}")

    except Exception as e:
        print("Servo error:", e)

# ==========================
# MAIN FLOW
# ==========================
@app.route("/start", methods=["GET"])
def start_process():
    try:
        print("Preparing for face capture...")
        time.sleep(2)   # give user time to position face

        print("Capturing fresh face...")
        face_image = capture_image("face.jpg")

        if face_image is None:
            return jsonify({"error": "Face capture failed"}), 500

        user, distance = recognize_face(face_image)

        if user is None:
            return jsonify({
                "status": "unknown",
                "message": "User not recognized"
            })

        confidence = float(round(1 - float(distance), 3))
        print("User recognized:", user)

        print("Now show waste object...")
        time.sleep(5)

        print("Capturing fresh waste...")
        waste_image = capture_image("waste.jpg")

        if waste_image is None:
            return jsonify({"error": "Waste capture failed"}), 500

        waste_type = classify_waste(waste_image)
        print("Waste detected:", waste_type)

        open_bin(waste_type)

        django_type = WASTE_TYPE_MAP.get(waste_type, waste_type)
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

# ==========================
# PREVIEW PAGE
# ==========================
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
        <img src="{CAMERA_IP}/capture">
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