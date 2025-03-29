from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        features_array = np.array([features])
        
        # Predict using the model
        prediction = model.predict(features_array)
        result = "Fraudulent Loan" if prediction[0] == 1 else "Legitimate Loan"
        
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
