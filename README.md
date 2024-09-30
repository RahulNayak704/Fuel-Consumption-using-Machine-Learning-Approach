# Fuel Consumption using Machine Learning Approach

## Description
This project predicts the fuel consumption of vehicles using machine learning algorithms.

## Installation
1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Run the Flask app: `python app.py`.

## Usage
Make a POST request to `/predict` with the following JSON:
```json
{
    "cylinders": 6,
    "displacement": 250,
    "horsepower": 130,
    "weight": 3000,
    "acceleration": 15,
    "model_year": 82
}
