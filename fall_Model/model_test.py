from joblib import dump, load
import numpy as np
model = load('/content/falldetect.joblib')
# /content/falldetect.joblib is the model from the fall dection

#  the nuber entered manualy is the eeg rading that is taken at rendom from the dataset and it gives the output as 1,0 for a fal and no fal senerio 

model.predict([[0.782950 ]])
