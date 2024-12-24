import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("customer_churn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()
        # Convert it to a DataFrame
        input_df = pd.DataFrame(input_data)
        # Predict probabilities
        predictions = model.predict(input_df).tolist()
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
