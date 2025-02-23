Step 1: Import Required Libraries

# Import necessary libraries
import numpy as np  # Used for numerical operations
import pandas as pd  # Used for handling tabular data
from pathlib import Path  # Used for handling file paths
from sklearn.metrics import confusion_matrix, classification_report  # Evaluation metrics
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.linear_model import LogisticRegression  # Logistic regression model
numpy helps with numerical computations.
pandas allows us to manipulate tabular data (CSV files).
Path makes it easier to work with file paths.
train_test_split is used to split the dataset into training and testing sets.
LogisticRegression is the classification model we use.
 Step 2: Load and Inspect Data

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("Resources/lending_data.csv")

# Display the first 5 rows of the dataset
df.head()
Reads the CSV file containing lending data.
Displays the first few rows for review.
Step 3: Define Features (X) and Labels (y)

# Separate the y variable (loan_status column)
y = df["loan_status"]

# Separate the X variable (all columns except loan_status)
X = df.drop(columns=["loan_status"])

# Review the y variable
print(y.value_counts())  # Check how many healthy/high-risk loans

# Review the X variable
print(X.head())  # Display first 5 rows of features
y (target variable) contains loan status:
0 = Healthy Loan
1 = High-Risk Loan
X (features) contains borrower attributes (income, credit score, etc.).
value_counts() helps check the distribution of 0s and 1s.
Step 4: Split Data into Training and Testing Sets

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Display dataset shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
train_test_split divides data:
80% training (X_train, y_train)
20% testing (X_test, y_test)
random_state=1 ensures consistent results.
 Step 5: Train the Logistic Regression Model


# Instantiate the Logistic Regression model with random_state=1
model = LogisticRegression(random_state=1)

# Train the model using the training data
model.fit(X_train, y_train)
Creates a logistic regression model.
Fits (trains) it using the training data.

Step 6: Make Predictions

# Make predictions using the testing dataset
y_pred = model.predict(X_test)

# Display first 10 predictions
print("Predicted labels:", y_pred[:10])
Uses the trained model to predict loan status on test data.
 Step 7: Evaluate the Model
Confusion Matrix

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
Shows how many predictions were correct and incorrect.
Classification Report

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
Gives accuracy, precision, and recall for both classes (0 and 1).
