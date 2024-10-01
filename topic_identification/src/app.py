from flask import Flask, request, jsonify
from .model import train_model  # You need to run this separately to get the model and vectorizer
import pickle

app = Flask(__name__)

# Load your model and vectorizer here (after training)
model = ...  # Load your trained model
vectorizer = ...  # Load your vectorizer

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return jsonify({'topic': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
