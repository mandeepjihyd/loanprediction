import pickle

file_path = r'C:\Users\DELL\mandeeproject\newmodel\loan_status_model.pkl'
try:
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully in test.")
except Exception as e:
    print(f"Test failed: {e}")
