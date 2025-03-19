import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate data
data = {
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.normal(35, 12, n_samples).clip(18, 80),  # Age between 18 and 80
    'Height': np.random.normal(170, 10, n_samples).clip(150, 200),  # Height in cm
    'Weight': np.random.normal(75, 20, n_samples).clip(40, 160),  # Weight in kg
    'SMOKE': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
    'FAVC': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),  # Frequent consumption of high caloric food
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate BMI
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# Define obesity based on BMI
def get_obesity_status(bmi):
    if bmi >= 30:
        return 'Yes'
    return 'No'

# Apply obesity classification
df['Obesity'] = df['BMI'].apply(get_obesity_status)

# Round numerical columns to 2 decimal places
df['Age'] = df['Age'].round(2)
df['Height'] = df['Height'].round(2)
df['Weight'] = df['Weight'].round(2)
df['BMI'] = df['BMI'].round(2)

# Save to CSV
df.to_csv('obesity_dataset.csv', index=False)
print("Dataset generated and saved to 'obesity_dataset.csv'")
print("\nSample distribution:")
print(df['Obesity'].value_counts())
