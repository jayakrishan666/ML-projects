from django.db import models

# Create your models here.

class ObesityData(models.Model):
    gender = models.CharField(max_length=10)
    age = models.FloatField()
    height = models.FloatField()
    weight = models.FloatField()
    smoke = models.BooleanField()
    favc = models.BooleanField()  # Frequent consumption of high caloric food
    obesity_level = models.CharField(max_length=50)

    def __str__(self):
        return f"Obesity Data #{self.id}"
