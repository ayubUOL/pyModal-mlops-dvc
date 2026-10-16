# Step 0: Import the tools we need
import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# MLflow setup: store all experiment results in a local database file
# and put this project's runs under one experiment name
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("house-price-project")

# Choose the model type from the command line, for example:
#   python train.py            -> linear (the default)
#   python train.py tree
#   python train.py forest
model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"

# Step 1: Read the data from houses.csv into a table (DataFrame)
data = pd.read_csv("houses.csv")

# Step 2: Choose features (X = what we know) and label (y = what we want to predict)
X = data[["rooms", "size_sqft"]]
y = data["price_lakh"]

# Step 3: Split the data: 80% for training, 20% for testing
# random_state=42 makes the split the same every time we run the script
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
print("Houses used for training:", len(X_train))
print("Houses used for testing:", len(X_test))

# Step 4: Create the model that matches the chosen type
if model_type == "linear":
    model = LinearRegression()
elif model_type == "tree":
    model = DecisionTreeRegressor(random_state=42)
elif model_type == "forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    sys.exit("Unknown model type. Use: linear, tree or forest")

# An MLflow "run" is one recorded experiment.
# Everything inside this block is tracked together.
with mlflow.start_run():

    # Step 4 (continued): Train the model on the training data
    model.fit(X_train, y_train)

    # Step 5: Use the model to predict prices for the test houses
    predictions = model.predict(X_test)

    # Show each test house: real price, the model's guess, and the difference
    print("\nTest results:")
    for i in range(len(X_test)):
        rooms = X_test.iloc[i]["rooms"]
        size = X_test.iloc[i]["size_sqft"]
        real_price = y_test.iloc[i]
        guess = predictions[i]
        error = abs(real_price - guess)
        print(f"Rooms: {rooms}, Size: {size} sqft, Real price: {real_price} lakh, "
              f"Guess: {guess:.1f} lakh, Off by: {error:.1f} lakh")

    # Step 6: Calculate the Mean Absolute Error (MAE)
    # MAE = the average size of the mistakes (lower is better)
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nModel type: {model_type}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} lakh")

    # Step 7: Tell MLflow what settings we used (parameters)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))

    # Step 8: Tell MLflow how well the model did (metric)
    mlflow.log_metric("mae", mae)

    # Save the score to a file so the CI quality gate can read it
    with open("metrics.json", "w") as f:
        json.dump({"mae": mae}, f)

    # Step 9: Save the model inside MLflow and register it by name
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="house-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

# Step 10: Also save the trained model to a file so predict.py can use it
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")