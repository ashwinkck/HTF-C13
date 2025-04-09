from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Correct model name and full path for Gemini Flash
genai.configure(api_key=os.getenv("AIzaSyBRO5MXtDqzfEeDAZif0wJ9ylaJMd67Z6M"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
chat_session = model.start_chat()

@app.route("/api/send_message", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message_text = data.get("message", "")
        if not message_text:
            return jsonify({"response": "No message provided"}), 400

        # Just send the plain user message
        response = chat_session.send_message(message_text)
        return jsonify({"response": response.text})

    except Exception as e:
        print("🔥 Error:", e)
        return jsonify({"response": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
