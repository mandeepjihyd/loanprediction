from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder=r"C:\Users\DELL\mandeeproject\newmodel\template")

# Define the path to the model file
file_path = r'C:\Users\DELL\mandeeproject\newmodel\loan_status_model.pkl'

# Initialize model variable
model = None

# Load model function
def load_model():
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:
            try:
                with open(file_path, 'rb') as file:
                    return pickle.load(file)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Model file is empty.")
    else:
        print("Model file not found at the specified path.")
    return None

# Load model at startup
model = load_model()
if model is not None:
    print("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is not None:
        try:
            # Collect data from form and convert safely
            data = []
            for value in request.form.values():
                try:
                    # Attempt to convert each form input to float
                    data.append(float(value))
                except ValueError:
                    return f"Invalid input '{value}'. Please enter numeric values only."

            # Convert to numpy array and reshape for a single sample
            data_array = np.array(data).reshape(1, -1)

            # Check if the array has the correct number of features
            if data_array.shape[1] != model.n_features_in_:
                return f"Incorrect number of inputs. Expected {model.n_features_in_}, got {data_array.shape[1]}."

            # Make prediction
            prediction = model.predict(data_array)
            output = 'Loan Approved' if prediction[0] == 1 else 'Loan Not Approved'
            return render_template('result.html', prediction_text=output)
        except Exception as e:
            return f"An error occurred during prediction: {e}"
    else:
        return "Model is not loaded. Please check the server logs."

if __name__ == "__main__":
    app.run(debug=True)
