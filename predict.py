# Step 1: Import the tools we need
import pandas as pd
import joblib

# Step 2: Load the trained model that train.py saved
model = joblib.load("model.pkl")

# Step 3: Describe the new house (the column names must match training)
new_house = pd.DataFrame({"rooms": [3], "size_sqft": [1350]})

# Step 4: Ask the model to predict the price (it returns a list, so we take [0])
price = model.predict(new_house)[0]

# Step 5: Print the result
print(f"3 rooms, 1350 sqft -> predicted price: {price:.1f} lakh")
