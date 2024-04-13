import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data
data_path = r"C:\Users\DELL\Downloads\processed_data_of_Loan_Prediction_Project_File_Final_Dataset_after_Data_Preprocessing.csv"
data = pd.read_csv(data_path)

# Assuming 'Dependents' is the problematic column
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)  # Convert '3+' to 3 and ensure the column is integer type

# Map other categorical data to integers
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 2, 'Urban': 1},
    'Loan_Status': {'Y': 1, 'N': 0}
}
data.replace(mappings, inplace=True)

# Standardize numeric columns
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
data[cols] = scaler.fit_transform(data[cols])

# Split data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save model
model_path = r'C:\Users\DELL\mandeeproject\newmodel\loan_status_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")

