import logging
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 
logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
class ZcSoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self,threshold=3):
        self.threshold=threshold
        
    def detect_outlier(self, df:pd.DataFrame) ->pd.DataFrame:
        logging.info("Detecting outlier using zscore method")
        z_scores=np.abs((df -df.mean())/df.std())
        outliers=z_scores>self.threshold
        logging.info(f"Outliers detected with z-score thresold :{self.threshold}")
        return outliers

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outlier(self, df: pd.DataFrame) ->pd.DataFrame:
        logging.info("Detecting outliers with IQR method.")
        q1= df.quantile(0.25)
        q3=df.quantile(0.75)
        IQR=q3-q1
        outliers= (df<(q1 -1.5 *IQR))| (df>(q3 + 1.5*IQR))
        logging.info("Outliers detected using IQR method.")
        return outliers

class OutlierDetector:
    def __init__(self,strategy:OutlierDetectionStrategy):
        self._strategy=strategy
    def set_strategy(self,strategy:OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy=strategy
    def detect_outlier(self,df:pd.DataFrame) ->pd.DataFrame:
        logging.info("Executed outlier strategy.")
        return self._strategy.detect_outlier(df)
    def handel_outliers(self,df:pd.DataFrame,method="remove",**kwargs) ->pd.DataFrame:
        outliers=self.detect_outlier(df)
        if method=="remove":
            logging.info("Removing outlier from data.")
            mask=(~outliers).all(axis=1)
            df_cleaned=df[mask]
        elif method=="cap":
            logging.info("Capping outliers from the data.")
            df_cleaned = df.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))

        else:
            logging.warning(f"Unknown method '{method}'. No outlier detected ")
            return df
        logging.info("Outlier handeling completed.")
        return df_cleaned
    def Visualise_outliers(self,df :pd.DataFrame,features:list):
        logging.info(f"Visuallizing outliers for features :{features}")
        for feature in features:
            plt.figure(figsize=(8,5))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxpllot of {feature}")
            plt.show()
        logging.info("Outlier visualisation completed.")