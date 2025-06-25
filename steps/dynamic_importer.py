import pandas as pd 
from zenml import step

@step
def dynamic_importer() ->str:
    data = {
        "Person ID": [1,5],
        "Gender": ["Male","Female"],
        "Age": [27,28],
        "Occupation": ["Software Engineer","Sales Representative"],
        "Sleep Duration": [6.1,5.9],
        "Quality of Sleep": [6,4],
        "Physical Activity Level": [42,30],
        "Stress Level": [6,8],
        "BMI Category": ["Overweight","Obese"],
        "Blood Pressure": ["126/83","140/90"],
        "Heart Rate": [77,85],
        "Daily Steps": [4200,3000]
        # "Sleep Disorder": [None,"Sleep Apnea"]
        }
    df=pd.DataFrame(data)
    json_data=df.to_json(orient="split")
    return json_data