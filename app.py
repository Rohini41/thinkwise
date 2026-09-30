import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
CHATBOT_TOPIC = os.getenv("CHATBOT_TOPIC", "ThinkWise AI")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from the .env file.")

app = Flask(__name__)
client = genai.Client(api_key=API_KEY)

system_prompt = (BASE_DIR / "chatbot_config").read_text(encoding="utf-8")


@app.get("/")
def home():
    return render_template("index.html", topic=CHATBOT_TOPIC)


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                max_output_tokens=1200,
            ),
        )

        answer = (response.text or "").strip()
        return jsonify({
            "answer": answer or "I couldn't generate a response. Please try again."
        })

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "The AI service could not process your request right now."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
