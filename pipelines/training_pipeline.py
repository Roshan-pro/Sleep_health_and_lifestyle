import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from steps.outlier_detection_step import outlier_detection_step
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.model_evaluator_step import model_evaluation_step
from steps.obj_to_int_step import Object_to_int_step
from steps.model_building_step import model_building_step
from steps.feature_enginnering_step import feature_engineering_step
from steps.handel_missing_values import handel_missing_values_step
from zenml import Model,pipeline,step
@pipeline(
    model=Model(
        name="Predicting_Sleep_Disorder"
    ),
)

def ml_pipeline():
    raw_data=data_ingestion_step(
        file_path=r"C:\Users\rk186\OneDrive\Desktop\Sleep_health_and_lifestyle\data\archive (5).zip"
    )
    filled_data=handel_missing_values_step(raw_data)
    object_to_int_=Object_to_int_step(filled_data)

    engineered_data=feature_engineering_step(
        object_to_int_,features=["Gender","Occupation","BMI Category","Blood Pressure"]
    )
    cleaned_data=outlier_detection_step(df=engineered_data,column_name=["Age","Sleep Duration","Quality of Sleep","Physical Activity Level","Stress Level","Heart Rate","Daily Steps"])
    x_train,x_test,y_train,y_test=data_splitter_step(cleaned_data,"Sleep Disorder")
    
    model=model_building_step(x_train=x_train,y_train=y_train)
    evaluation_metrics=model_evaluation_step(model,x_test,y_test)
    return model 
