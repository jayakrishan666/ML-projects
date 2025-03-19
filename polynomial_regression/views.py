from django.shortcuts import render
from django.views.generic import FormView
from logistic_regression.forms import ObesityDataForm
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

class PredictView(FormView):
    template_name = 'predict.html'
    form_class = ObesityDataForm
    success_url = '.'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Obesity Prediction using Polynomial Regression'
        return context

    def form_valid(self, form):
        # Get the form data
        data = form.cleaned_data
        prediction = None
        confidence = None
        error_message = None
        graph_image = None
        
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
                model = joblib.load('polynomial_regression/model/poly_model.pkl')
                scaler = joblib.load('polynomial_regression/model/scaler.pkl')
                poly = joblib.load('polynomial_regression/model/poly_features.pkl')
                
                # Scale the features
                features_scaled = scaler.transform(features)
                
                # Create polynomial features
                features_poly = poly.transform(features_scaled)
                
                # Make prediction
                prediction_numeric = model.predict(features_poly)[0]
                prediction = self._get_obesity_level(prediction_numeric)
                
                # Add BMI to the prediction
                bmi_rounded = round(bmi, 2)
                prediction = f"{prediction} (BMI: {bmi_rounded})"
                
                # Calculate confidence/score
                confidence = f"Prediction Score: {abs(prediction_numeric - round(prediction_numeric)):.2f}"
                
                # Generate prediction graph
                graph_image = self._generate_prediction_graph(bmi, prediction_numeric, model, scaler, poly)
                
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
            'title': 'Obesity Prediction using Polynomial Regression',
            'prediction': prediction if not error_message else None,
            'confidence': confidence if not error_message else None,
            'error_message': error_message,
            'graph_image': graph_image
        }
        
        return render(self.request, self.template_name, context)

    def _get_obesity_level(self, numeric_prediction):
        levels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
                 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 
                 'Obesity Type III']
        index = int(round(numeric_prediction))
        index = max(0, min(index, len(levels) - 1))
        return levels[index]

    def _generate_prediction_graph(self, bmi, prediction, model, scaler, poly):
        try:
            # Set dark theme style
            plt.style.use('dark_background')
            
            # Create BMI range for visualization
            bmi_range = np.linspace(10, 50, 100).reshape(-1, 1)  # BMI from 10 to 50
            
            # Scale and transform the BMI range
            bmi_scaled = scaler.transform(bmi_range)
            bmi_poly = poly.transform(bmi_scaled)
            
            # Get predictions for the BMI range
            predictions = model.predict(bmi_poly)
            
            # Create the plot with dark theme
            plt.figure(figsize=(12, 7), facecolor='#242424')
            ax = plt.gca()
            ax.set_facecolor('#242424')
            
            # Plot main curve and point
            plt.plot(bmi_range, predictions, color='#6c5ce7', linewidth=2.5, label='Prediction curve')
            plt.scatter([bmi], [prediction], color='#ff6b6b', s=150, label='Your BMI', zorder=5)
            
            # Add labels and title with custom colors
            plt.xlabel('BMI', color='white', fontsize=12)
            plt.ylabel('Obesity Level', color='white', fontsize=12)
            plt.title('Polynomial Regression Prediction', color='white', fontsize=14, pad=20)
            
            # Customize grid
            plt.grid(True, linestyle='--', alpha=0.2)
            
            # Add horizontal lines for obesity levels
            for i in range(7):
                plt.axhline(y=i, color='gray', linestyle='--', alpha=0.2)
                plt.text(52, i, self._get_obesity_level(i), fontsize=10, 
                        color='white', verticalalignment='center')
            
            # Customize ticks
            plt.tick_params(colors='white')
            
            # Add legend with custom style
            legend = plt.legend(facecolor='#242424', edgecolor='#333333')
            plt.setp(legend.get_texts(), color='white')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot to BytesIO object with transparent background
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='#242424', edgecolor='none', 
                       bbox_inches='tight', dpi=100, pad_inches=0.2)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Encode the image to base64
            graphic = base64.b64encode(image_png)
            graph_url = graphic.decode('utf-8')
            
            return f"data:image/png;base64,{graph_url}"
            
        except Exception as e:
            print(f"Error generating graph: {str(e)}")
            return None
