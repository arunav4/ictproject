from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model, label encoders, and feature columns
best_rf_model = joblib.load('best_rf_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Define the categorical columns that were encoded
categorical_cols = ['type', 'director', 'cast', 'country', 'rating', 'listed_in']
# Exclude 'type' as it is the target variable for prediction
categorical_cols_for_input = [col for col in categorical_cols if col != 'type']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'director': request.form['director'],
        'cast': request.form['cast'],
        'country': request.form['country'],
        'release_year': int(request.form['release_year']),
        'rating': request.form['rating'],
        'listed_in': request.form['listed_in'],
        'month_added': int(request.form['month_added']),
        'year_added': int(request.form['year_added']),
        'duration_minutes': int(request.form['duration_minutes']),
        'num_seasons': int(request.form['num_seasons'])
    }

    # Create a DataFrame for the new data point
    new_df = pd.DataFrame([data])

    # Apply the same label encoding transformations
    for col in categorical_cols_for_input:
        if col in new_df.columns and col in label_encoders:
            try:
                new_df[col] = label_encoders[col].transform(new_df[col])
            except ValueError:
                # Handle unseen labels: assign a default value (e.g., 0 for 'Unknown' or an average)
                # For simplicity, we assign 0, which corresponds to 'Unknown' in most of our encoders
                # A more robust solution might involve mapping to a specific 'unseen' category or using a different encoding strategy
                new_df[col] = 0
                print(f"Warning: Unseen label for column '{col}' encountered. Assigned 0.")

    # Ensure the columns are in the same order as the training data
    new_df = new_df[feature_columns]

    # Predict using the best model
    prediction_encoded = best_rf_model.predict(new_df)

    # Inverse transform the prediction to get the original label
    predicted_type = label_encoders['type'].inverse_transform(prediction_encoded)

    return render_template('result.html', prediction=predicted_type[0])

if __name__ == '__main__':
    # You can run this directly in a Colab environment or save as app.py and run locally.
    # In Colab, you might need to use ngrok for external access if running outside localtunnel.
    # For local execution, run `python app.py` in your terminal.
    app.run(host='0.0.0.0', port=5000)
