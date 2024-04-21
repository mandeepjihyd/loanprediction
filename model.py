import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data
data_path = r"C:\Users\DELL\Downloads\processed_data_of_Loan_Prediction_Project_File_Final_Dataset_after_Data_Preprocessing.csv"
data = pd.read_csv(data_path)

# Convert '3+' to 3 and ensure the column is integer type
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

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

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Create a RandomForest Classifier
rf = RandomForestClassifier(random_state=42)

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Use the best estimator to make predictions
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with best parameters: {accuracy}")

# Save the best model
model_path = 'model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)
print("Best model saved successfully.")
