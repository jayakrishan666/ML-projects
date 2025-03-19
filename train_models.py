import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create model directories if they don't exist
os.makedirs('logistic_regression/model', exist_ok=True)
os.makedirs('knn/model', exist_ok=True)
os.makedirs('polynomial_regression/model', exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv('obesity_data.csv')

# Prepare features and target
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Convert categorical variables to numeric
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, lr_pred))

# Save Logistic Regression model and scaler
joblib.dump(lr_model, 'logistic_regression/model/logistic_model.pkl')
joblib.dump(scaler, 'logistic_regression/model/scaler.pkl')

# Train KNN
print("\nTraining KNN model...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_accuracy)
print("\nClassification Report for KNN:")
print(classification_report(y_test, knn_pred))

# Save KNN model and scaler
joblib.dump(knn_model, 'knn/model/knn_model.pkl')
joblib.dump(scaler, 'knn/model/scaler.pkl')

# Train Polynomial Regression
print("\nTraining Polynomial Regression model...")
# Convert target to numeric
le_target = LabelEncoder()
y_train_numeric = le_target.fit_transform(y_train)
y_test_numeric = le_target.transform(y_test)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_numeric)
poly_pred_numeric = poly_model.predict(X_test_poly)
poly_pred = le_target.inverse_transform(np.round(poly_pred_numeric).astype(int).clip(0, len(le_target.classes_)-1))
poly_accuracy = accuracy_score(y_test, poly_pred)
print("Polynomial Regression Accuracy:", poly_accuracy)
print("\nClassification Report for Polynomial Regression:")
print(classification_report(y_test, poly_pred))

# Save Polynomial Regression model, scaler, and polynomial features
joblib.dump(poly_model, 'polynomial_regression/model/poly_model.pkl')
joblib.dump(scaler, 'polynomial_regression/model/scaler.pkl')
joblib.dump(poly, 'polynomial_regression/model/poly_features.pkl')

print("\nAll models have been trained and saved successfully!")
