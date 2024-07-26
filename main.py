from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


# Load the scaler and model with error handling
try:
    scaler = pickle.load(open('D:/Bhavesh/machine_learning_project/Credit_Card_Default_Prediction/model/standardscaler.pkl','rb'))
    model = pickle.load(open('D:/Bhavesh/machine_learning_project/Credit_Card_Default_Prediction/model/modelForPrediction.pkl', "rb"))
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        try:
            # Log request form data
            print(request.form)

            # Validate and extract data from form
            name = request.form.get('Name')
            age = request.form.get('Age')
            education = request.form.get('Education')
            mobile_number = request.form.get('Mobile_number')
            marital_status = request.form.get('Marital_Status')
            credit_limit = request.form.get('Credit_Limit')
            payment_history = request.form.get('Payment_History')
            limit_balance = request.form.get('Limit_Balance')

            # Ensure all required fields are present
            if not all([name, age, education, mobile_number, marital_status, credit_limit, payment_history, limit_balance]):
                result = "Error: Missing form fields."
                return render_template('single_prediction.html', result=result)

            # Convert fields to appropriate types
            age = int(age)
            credit_limit = float(credit_limit)
            limit_balance = float(limit_balance)

            # Encode categorical variables
            education_encoded = encode_education(education)
            marital_status_encoded = encode_marital_status(marital_status)
            payment_history_encoded = encode_payment_history(payment_history)

            # Example: Including placeholders for all 23 features
            # This should be replaced with actual data processing to include all necessary features
            features = [age, education_encoded, marital_status_encoded, credit_limit, payment_history_encoded, limit_balance]

            # Assuming you need 23 features, pad with zeros or provide real data
            while len(features) < 23:
                features.append(0)

            features = np.array(features).reshape(1, -1)

            # Create feature array for prediction
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                result = 'Approval'
            else:
                result = 'Reject'
        except Exception as e:
            result = f"Error in processing input data: {e}"

        return render_template('single_prediction.html', result=result)
    else:
        return render_template('index.html')

# Example encoding functions
def encode_education(education):
    encoding_dict = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    return encoding_dict.get(education, -1)  # Default to -1 if not found

def encode_marital_status(status):
    encoding_dict = {'Single': 0, 'Married': 1, 'Divorced': 2}
    return encoding_dict.get(status, -1)  # Default to -1 if not found

def encode_payment_history(history):
    encoding_dict = {'Good': 0, 'Average': 1, 'Poor': 2}
    return encoding_dict.get(history, -1)  # Default to -1 if not found

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=8080)


# can you run this file and search http://localhost:8080/predictdata 