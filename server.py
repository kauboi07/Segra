from flask import Flask
from api.process import process_api

app = Flask(__name__)

app.register_blueprint(process_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
