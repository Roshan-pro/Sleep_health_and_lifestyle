import pandas as pd 
from typing import Tuple
import logging
from src.data_splitter import DataSpliter,SimpleTrainTestSplit
from zenml import step
@step
def data_splitter_step(df:pd.DataFrame,target_col:str)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logging.info("Running data_splitter_step...")
    logging.info(f"Data present before splitting {df.columns}")
    splitter=DataSpliter(strategy=SimpleTrainTestSplit())
    x_train,x_test,y_train,y_test=splitter.split(df,target_col)
    return x_train,x_test,y_train,y_test
