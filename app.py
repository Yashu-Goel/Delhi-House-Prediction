from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


# Load model, scaler, and expected columns
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Load the dataset to extract dropdown values
df = pd.read_csv('MagicBricks.csv')  # Make sure this file is in the same folder

# Extract all the unique values needed for form dropdowns
bathroom_options = sorted(df['Bathroom'].dropna().unique().astype(int))
parking_options = sorted(df['Parking'].dropna().unique().astype(int))
type_options = sorted(df['Type'].dropna().unique())
furnishing_options = sorted(df['Furnishing'].dropna().unique())
status_options = sorted(df['Status'].dropna().unique())
transaction_options = sorted(df['Transaction'].dropna().unique())
locality_options = sorted(df['Locality'].dropna().unique())

# Collect all dropdown values
dropdown_data = {
    'Bathroom': bathroom_options,
    'Parking': parking_options,
    'Type': type_options,
    'Furnishing': furnishing_options,
    'Status': status_options,
    'Transaction': transaction_options,
    'Locality': locality_options
}


@app.route('/')
def home():
    return render_template('index.html', dropdowns=dropdown_data)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    input_dict = {
        'Area': float(data['TotalArea']),
        'BHK': int(data['BHK']),
        'Bathroom': int(data['Bathroom']),
        'Per_Sqft': float(data['PerSqft']),
        'Furnishing': data['Furnishing'],
        'Parking': int(data['Parking']),
        'Status': data['Status'],
        'Type': data['Type'],
        'Transaction': data['Transaction'],
        'Location': data['Locality']
    }

    df_input = pd.DataFrame([input_dict])

    # Label encoding
    df_input.replace({'Furnishing': {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}}, inplace=True)
    df_input.replace({'Parking': {'No': 0, 'Yes': 1}}, inplace=True)
    df_input.replace({'Status': {'Ready_to_move': 0, 'Almost_ready': 1}}, inplace=True)
    df_input.replace({'Type': {'Builder_Floor': 0, 'Apartment': 1}}, inplace=True)
    df_input.replace({'Transaction': {'Resale': 0, 'New_Property': 1}}, inplace=True)

    # Scale numerical features
    numeric = ['Area', 'BHK', 'Bathroom', 'Per_Sqft']
    df_input[numeric] = scaler.transform(df_input[numeric])

    # One-hot encoding for Location
    location_dummies = pd.get_dummies(df_input['Location'], drop_first=True, prefix='Location')
    df_input.drop(columns=['Location'], inplace=True)
    df_input = pd.concat([df_input, location_dummies], axis=1)

    # Add any missing columns
    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_columns]  # Ensure correct column order

    # Predict
    prediction = model.predict(df_input)[0]

    return render_template('index.html', prediction=round(prediction, 2), dropdowns=dropdown_data)

if __name__ == '__main__':
    app.run(debug=True)
