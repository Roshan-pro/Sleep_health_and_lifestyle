import logging
from abc import ABC,abstractmethod
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")

class ObjectToIntegerStrategy(ABC):
    @abstractmethod
    def apply(self,df:pd.DataFrame):
        pass
class SimpleObjectToInteger(ObjectToIntegerStrategy):
    def __init__(self):
        self.encoder=LabelEncoder()
    def apply(self, df:pd.DataFrame):
        object_col=df.select_dtypes(include=["object"]).columns
        logging.info(f"Converting Objects to integer in columns:{object_col.to_list()}")
        df_transformed=df.copy()
        for col in object_col:
            df_transformed[col]=self.encoder.fit_transform(df_transformed[col])
        return df_transformed
class ObjectToIntegerConverter:
    def __init__(self,strategy:ObjectToIntegerStrategy):
        self.strategy=strategy
    def apply(self,df:pd.DataFrame):
        return self.strategy.apply(df)