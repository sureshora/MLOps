# deploy_model.py

from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the correct path to the 'model.pkl' file in the 'model' directory
model_path = os.path.join(script_dir, '..', 'model', 'model.pkl')

# Load the trained model from the correct path
model = joblib.load(model_path)

@app.route('/')
def home():
    return 'Model Deployment API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json

        # Perform inference using the loaded model
        prediction = model.predict([data['input_features']])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
