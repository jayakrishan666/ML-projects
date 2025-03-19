from django import forms
from .models import ObesityData

class ObesityDataForm(forms.ModelForm):
    class Meta:
        model = ObesityData
        exclude = ['obesity_level']
        widgets = {
            'gender': forms.Select(choices=[('Male', 'Male'), ('Female', 'Female')]),
            'family_history': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')]),
            'favc': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')]),
            'caec': forms.Select(choices=[
                ('Never', 'Never'),
                ('Sometimes', 'Sometimes'),
                ('Frequently', 'Frequently'),
                ('Always', 'Always')
            ]),
            'smoke': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')]),
            'scc': forms.RadioSelect(choices=[(True, 'Yes'), (False, 'No')]),
            'calc': forms.Select(choices=[
                ('No', 'No'),
                ('Sometimes', 'Sometimes'),
                ('Frequently', 'Frequently'),
                ('Always', 'Always')
            ]),
            'mtrans': forms.Select(choices=[
                ('Automobile', 'Automobile'),
                ('Bike', 'Bike'),
                ('Motorbike', 'Motorbike'),
                ('Public_Transportation', 'Public Transportation'),
                ('Walking', 'Walking')
            ])
        }
