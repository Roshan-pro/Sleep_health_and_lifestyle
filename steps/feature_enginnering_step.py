import pandas as pd 
import logging
from src.feature_engineering import StandardScaling,MinMaxScaling,OneHotEncodingScaling,FeatureEngineering
from zenml import step
@step
def feature_engineering_step(df:pd.DataFrame,features:list[str]):
    feature_eng=FeatureEngineering(StandardScaling(features))
    transformed_df=feature_eng.apply_transformation(df)
    return transformed_df