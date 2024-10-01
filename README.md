# Fuel Consumption using Machine Learning Approach

## Description
This project predicts the fuel consumption of vehicles using machine learning algorithms.

## Installation
1. Clone the repository.
2. Navigate to the project directory: `cd fuel-consumption-prediction`.
3. Install the dependencies: `pip install -r requirements.txt`.
4. Run the data preprocessing script: `python data_preprocessing.py`.
5. Train the machine learning models: `python model_training.py`.
6. Run the Flask app: `python app.py`.

The application will be running on http://127.0.0.1:5000/


Usage
Predict Fuel Consumption
To predict the fuel consumption (MPG), make a POST request to /predict with the following JSON structure:

POST /predict
Content-Type: application/json
{
  "cylinders": 6,
  "displacement": 250,
  "horsepower": 130,
  "weight": 3000,
  "acceleration": 15,
  "model_year": 82
}

The response will return the predicted miles per gallon (MPG):

{
    "mpg_prediction": 18.5
}
