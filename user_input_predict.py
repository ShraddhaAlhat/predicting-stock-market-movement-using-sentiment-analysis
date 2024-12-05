import pandas as pd
import joblib  # Make sure joblib is imported

# Load the trained model and scaler from the saved files
model = joblib.load('best_trained_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of features for which we will take user input
features = ['positive', 'neutral', 'negative', 'compound', 'upvotes', 'num_comments', 'Open', 'High', 'Low', 
            'Close', 'Adj Close', 'Volume', 'EMA_50', 'SMA_5', 'RSI_14', 'Bollinger_MA', 'Bollinger_STD', 
            'Bollinger_Upper', 'Bollinger_Lower', 'LogMomentum']

# Take input from the user for each feature
user_input = []
print("Please enter the values for the following features:")

for feature in features:
    while True:
        try:
            # Prompt user for input for each feature
            value = float(input(f"{feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print(f"Invalid input. Please enter a numerical value for {feature}.")

# Create a DataFrame from the user's input
new_data = pd.DataFrame([user_input], columns=features)

# Print all features and their values entered by the user
print("\nYou entered the following feature values:")
for feature, value in zip(features, new_data.iloc[0]):
    print(f"{feature}: {value}")

# Standardize the new data using the same scaler (MinMaxScaler) that was used to scale the training data.
new_data_scaled = scaler.transform(new_data)

# Use the trained RandomForest model to predict the momentum for the new data point.
predicted_momentum = model.predict(new_data_scaled)

# Print the predicted momentum for the new data point.
print("\nPredicted Momentum (1 for Up, 0 for Down):", predicted_momentum[0])
