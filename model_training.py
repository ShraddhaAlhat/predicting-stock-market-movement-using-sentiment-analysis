# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 

# Load the dataset
data = pd.read_csv('stock_analysis.csv')

# Create the target column 'momentum'
data['momentum'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop the last row (it has no target value for 'momentum')
data = data[:-1]

# Select features and target
features = ['positive', 'neutral', 'negative', 'compound', 'upvotes', 'num_comments',
            'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA_50', 'SMA_5',
            'RSI_14', 'Bollinger_MA', 'Bollinger_STD', 'Bollinger_Upper', 'Bollinger_Lower', 'LogMomentum']
X = data[features]
y = data['momentum']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the feature data to scale between 0 and 1
X_normalized = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define the RandomForestClassifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

# Define the parameter grid for RandomizedSearchCV (smaller grid for faster results)
param_grid = {
    'n_estimators': [100, 200],  # Fewer trees to reduce computation time
    'max_depth': [10, 20],  # Fewer depth values to check
    'min_samples_split': [2, 5],  # Fewer values for splitting
    'min_samples_leaf': [1, 2],  # Fewer leaf nodes
    'max_features': ['sqrt', 'log2']  # Correct options for feature selection
}

# Implement RandomizedSearchCV to tune hyperparameters
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=2)

# Fit the RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Print the best hyperparameters found by RandomizedSearchCV
print(f"Best Parameters: {random_search.best_params_}")

# Get the best model from random search
best_model_random = random_search.best_estimator_

# Predict on the test set
y_pred = best_model_random.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and the scaler to files
joblib.dump(best_model_random, 'best_trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Best model and scaler saved successfully!")
