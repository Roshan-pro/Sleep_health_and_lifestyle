import logging
from abc import ABC,abstractmethod
import pandas as pd 

logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")
class MissingValueMethodStrategy(ABC):
    @abstractmethod
    def handel(self,df:pd.DataFrame) ->pd.DataFrame:
        pass
class FillMissingValueStrategy(MissingValueMethodStrategy):
    def __init__(self,method="mean",FillValue=None):
        self.method=method
        self.FillValue=FillValue
    def handel(self, df:pd.DataFrame):
        logging.info(f"Filling missing value with {self.method}")
        df_cleaned=df.copy()
        if self.method=="mean":
            numeric_col=df_cleaned.select_dtypes(include="int").columns
            df_cleaned[numeric_col]=df_cleaned[numeric_col].fillna(df_cleaned[numeric_col].mean())
        elif self.method=="median":
            numeric_col=df_cleaned.select_dtypes(include="int").columns
            df_cleaned[numeric_col]=df_cleaned[numeric_col].fillna(df_cleaned[numeric_col].median())
        elif self.method=="mode":
            for col in df_cleaned.columns:
                    df_cleaned[col]=df_cleaned[col].fillna(df_cleaned[col].mode())
        elif self.method=="constant":
            df_cleaned=df_cleaned.fillna(self.FillValue)
        else:
            logging.warning(f"Unknown method '{self.method}'")
        logging.info("Missing value filled")
        return df_cleaned
class MissingValueHandeler:
    def __init__(self,strategy : MissingValueMethodStrategy):
        self.strategy=strategy
    def handel_missing_values(self,df:pd.DataFrame):
        return self.strategy.handel(df)