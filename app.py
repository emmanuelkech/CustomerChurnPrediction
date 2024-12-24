from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("customer_churn_model.pkl")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle POST request
        data = request.get_json()
        return jsonify({"received_data": data})
    else:
        # Handle GET request
        return "This is the predict endpoint. Use POST to submit data."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


