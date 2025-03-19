from django import forms
from .models import ObesityData

class ObesityDataForm(forms.ModelForm):
    class Meta:
        model = ObesityData
        exclude = ['obesity_level']
        widgets = {
            'gender': forms.Select(choices=[('Male', 'Male'), ('Female', 'Female')]),
            'age': forms.NumberInput(attrs={'min': 18, 'max': 80}),
            'height': forms.NumberInput(attrs={'min': 150, 'max': 200, 'placeholder': 'Height in cm'}),
            'weight': forms.NumberInput(attrs={'min': 40, 'max': 160, 'placeholder': 'Weight in kg'}),
            'smoke': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')]),
            'favc': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')])
        }
        labels = {
            'favc': 'Frequent consumption of high caloric food'
        }
        help_texts = {
            'height': 'Enter height in centimeters (150-200)',
            'weight': 'Enter weight in kilograms (40-160)',
            'age': 'Enter age in years (18-80)',
            'smoke': 'Do you smoke?',
            'favc': 'Do you frequently consume high caloric food?'
        }
