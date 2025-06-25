import logging
import pandas as pd 
from src.obj_to_int import SimpleObjectToInteger,ObjectToIntegerConverter
from zenml import step
@step
def Object_to_int_step(df:pd.DataFrame):
    obj_int=ObjectToIntegerConverter(SimpleObjectToInteger())
    tranformd_data=obj_int.apply(df)
    return tranformd_data