from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Define the path to the model file
file_path = 'loan_status_model.pkl'  # Relative path

# Load model function
def load_model():
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_model()
if model:
    print("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            # Process form data and predict
            data = [float(value) for value in request.form.values()]
            data_array = np.array(data).reshape(1, -1)

            if data_array.shape[1] != model.n_features_in_:
                return f"Incorrect number of inputs. Expected {model.n_features_in_}, got {data_array.shape[1]}."

            prediction = model.predict(data_array)
            output = 'Loan Approved' if prediction[0] == 1 else 'Loan Not Approved'
            return render_template('result.html', prediction_text=output)
        except ValueError as e:
            return f"Invalid input. Please enter numeric values only. Error: {e}"
        except Exception as e:
            return f"An error occurred during prediction: {e}"
    else:
        return "Model is not loaded. Please check the server logs."
        
if __name__ == "__main__":
    app.run(debug=True)

