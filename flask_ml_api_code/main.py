import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('DiabetesModel.pkl', "rb") as model_file:
    model = pickle.load(model_file)

with open('Scaler.pkl', "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return "Diabetes Model API Prediction App is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get the JSON data from the api request
        data = request.get_json()

        input_data = pd.DataFrame([data])

        # check if input is provide
        if not data:
            return jsonify({"error": "Input Data Not Provided"}), 400

        # validate input columns
        required_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                         "Insulin","BMI","DiabetesPedigreeFunction","Age"]

        if not all(col in input_data.columns for col in required_cols):
            return jsonify({"error": f"Missing Required Columns. Required Columns: {required_cols}"}), 400

        # scale the data
        scaled_data = scaler.transform(input_data)

        # make predictions
        prediction = model.predict(scaled_data)

        # response
        response = {
            "prediction": "Diabetes" if prediction[0] == 1 else "Not Diabetes"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)