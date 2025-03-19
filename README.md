# Obesity Prediction Web Application

A Django-based web application that uses three different machine learning algorithms to predict obesity levels based on physical and lifestyle characteristics.

## Features

- Three different prediction models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Polynomial Regression
- User-friendly web interface
- Real-time predictions
- Confidence scores for predictions

## Installation

1. Clone the repository
2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

## Training the Models

1. Place your obesity dataset as `obesity_data.csv` in the project root directory
2. Run the training script:
```bash
python train_models.py
```

The script will create and train three different models and save them in their respective app directories.

## Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Visit http://localhost:8000 in your web browser

## Dataset Requirements

The `obesity_data.csv` file should contain the following columns:
- Gender
- Age
- Height
- Weight
- family_history_with_overweight
- FAVC (Frequent consumption of high caloric food)
- FCVC (Frequency of consumption of vegetables)
- NCP (Number of main meals)
- CAEC (Consumption of food between meals)
- SMOKE
- CH2O (Consumption of water daily)
- SCC (Calories consumption monitoring)
- FAF (Physical activity frequency)
- TUE (Time using technology devices)
- CALC (Consumption of alcohol)
- MTRANS (Transportation used)
- NObeyesdad (Obesity level - Target variable)

## Model Performance

The training script will output performance metrics for each model:
- Logistic Regression: Classification accuracy
- KNN: Classification accuracy
- Polynomial Regression: R² score
