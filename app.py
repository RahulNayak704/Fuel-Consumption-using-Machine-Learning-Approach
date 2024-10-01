from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Random Forest model
model = joblib.load('models/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (expects JSON)
    data = request.get_json()
    
    # Extract features from the input JSON
    features = [data['cylinders'], data['displacement'], data['horsepower'],
                data['weight'], data['acceleration'], data['model_year']]
    
    # Convert to numpy array and reshape for prediction
    features_array = np.array([features])
    
    # Make the prediction
    prediction = model.predict(features_array)
    
    # Return the predicted MPG as a JSON response
    return jsonify({'mpg_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
