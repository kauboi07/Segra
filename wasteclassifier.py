import base64
import os
from openai import OpenAI
from PIL import Image

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_image(image_path):
    img = Image.open(image_path)

    width, height = img.size

    left = width * 0.2
    top = height * 0.2
    right = width * 0.8
    bottom = height * 0.8

    img = img.crop((left, top, right, bottom))

    img = img.resize((512, 512))

    img.save(image_path)

def classify_waste(image_path):
    preprocess_image(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict waste classification AI .\n"
                    "that classifies the object into EXACTLY ONE of these:\n"
                    "plastic\nmetal\norganic\newaste\n\n"
                    "Respond with ONLY the word.\n"
                    "If unsure, choose the closest category.\n"
                    "Never say I don't know."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this waste item."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=5
    )

    result = response.choices[0].message.content.strip().lower()

    #placeholder response
    allowed = ["plastic", "metal", "organic", "ewaste"]
    if result not in allowed:
        return "plastic"

    return result