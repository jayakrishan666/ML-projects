from django.db import models

class ObesityData(models.Model):
    gender = models.CharField(max_length=10)
    age = models.FloatField()
    height = models.FloatField()
    weight = models.FloatField()
    family_history = models.BooleanField()
    favc = models.BooleanField()  # Frequent consumption of high caloric food
    fcvc = models.FloatField()    # Frequency of consumption of vegetables
    ncp = models.IntegerField()   # Number of main meals
    caec = models.CharField(max_length=20)  # Consumption of food between meals
    smoke = models.BooleanField()
    ch2o = models.FloatField()    # Consumption of water daily
    scc = models.BooleanField()   # Calories consumption monitoring
    faf = models.FloatField()     # Physical activity frequency
    tue = models.FloatField()     # Time using technology devices
    calc = models.CharField(max_length=20)  # Consumption of alcohol
    mtrans = models.CharField(max_length=20)  # Transportation used
    obesity_level = models.CharField(max_length=50)

    def __str__(self):
        return f"Obesity Data #{self.id}"
