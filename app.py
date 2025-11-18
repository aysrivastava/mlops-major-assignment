from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import os

app = Flask(__name__)

# Load the trained model
print("Loading trained model...")
model = joblib.load('savedmodel.pth')
print("Model loaded successfully!")

# Load dataset to get sample data structure
data = fetch_olivetti_faces(shuffle=True, random_state=42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # For demonstration, we'll use a random sample from the dataset
        # In a real scenario, this would process uploaded image data
        random_index = np.random.randint(0, len(data.data))
        sample = data.data[random_index].reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(sample)[0]
        actual_label = data.target[random_index]
        
        return jsonify({
            'prediction': int(prediction),
            'actual': int(actual_label),
            'correct': prediction == actual_label,
            'sample_index': random_index
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
