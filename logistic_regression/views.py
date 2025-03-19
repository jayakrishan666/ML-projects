from django.shortcuts import render
from django.views.generic import FormView
from .forms import ObesityDataForm
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class PredictView(FormView):
    template_name = 'predict.html'
    form_class = ObesityDataForm
    success_url = '.'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Obesity Prediction using Logistic Regression'
        return context

    def form_valid(self, form):
        # Get the form data
        data = form.cleaned_data
        prediction = None
        confidence = None
        error_message = None
        
        try:
            # Validate height and weight
            if data['height'] <= 0:
                raise ValueError("Height must be greater than 0")
            if data['weight'] <= 0:
                raise ValueError("Weight must be greater than 0")
            
            # Calculate BMI
            height_m = data['height'] / 100  # Convert cm to meters
            weight_kg = data['weight']
            bmi = weight_kg / (height_m * height_m)
            
            # Validate BMI range
            if bmi <= 0 or bmi > 100:  # Reasonable BMI range check
                raise ValueError("Calculated BMI is outside reasonable range")
            
            # Prepare feature (BMI only)
            features = np.array([[bmi]])
            
            # Load the model and make prediction
            try:
                model = joblib.load('logistic_regression/model/logistic_model.pkl')
                scaler = joblib.load('logistic_regression/model/scaler.pkl')
                
                # Scale the features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction_num = model.predict(features_scaled)[0]
                # Convert to Yes/No
                prediction = "Yes" if prediction_num >= 1 else "No"  # Yes if BMI indicates overweight or above
                
                # Get probability scores
                probabilities = model.predict_proba(features_scaled)[0]
                prob_percentage = max(probabilities) * 100
                confidence = f"Confidence: {prob_percentage:.2f}%"
                
                # Add BMI to the context
                bmi_rounded = round(bmi, 2)
                prediction = f"Obesity: {prediction} (BMI: {bmi_rounded})"
                
            except FileNotFoundError:
                error_message = "Model not trained yet. Please train the model first."
            except Exception as e:
                error_message = f"Error during prediction: {str(e)}"
                
        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
        
        context = {
            'form': form,
            'title': 'Obesity Prediction using Logistic Regression',
            'prediction': prediction if not error_message else None,
            'confidence': confidence if not error_message else None,
            'error_message': error_message
        }
        
        return render(self.request, self.template_name, context)
