from flask import Flask, request, jsonify
import joblib
import re

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 Spam Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    message_clean = preprocess(message)
    features = vectorizer.transform([message_clean])
    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": "Spam" if prediction == 1 else "Not Spam"
    })

if __name__ == "__main__":
    app.run(debug=True)
