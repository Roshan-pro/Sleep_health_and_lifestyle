import logging
from abc import ABC,abstractmethod
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler
logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df : pd.DataFrame) ->pd.DataFrame:
        pass
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
    def apply_transformation(self,df : pd.DataFrame)->pd.DataFrame:
        logging.info(f"Applying log transformation in features: {self.features}")
        df_transformed=df.copy()
        for feature in self.features:
            df_transformed[feature]=np.log1p(df[feature])
        logging.info("Log tranformation completed")
        return df_transformed
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
        self.scaler=StandardScaler()
    def apply_transformation(self, df: pd.DataFrame) ->pd.DataFrame:
        logging.info(f"Applying standard scaling to features {self.features}")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Standard scalling completd.")
        return df_transformed
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features,features_range=(0,1)):
        self.features=features
        self.scaler=MinMaxScaler(feature_range=features_range)
    def apply_transformation(self, df: pd.DataFrame) ->pd.DataFrame:
        logging.info(f"Applying MinMax scaling to features {self.features}")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Min Max scalling completd.")
        return df_transformed
class OneHotEncodingScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features=features
        self.encoder=OneHotEncoder(sparse_output=False,drop="first")
    def apply_transformation(self, df: pd.DataFrame) ->pd.DataFrame:
        logging.info(f"Applying one hot encoder  to features {self.features}")
        df_transformed=df.copy()
        encoded_df=pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
            index=df.index
            )
        df_transformed=df_transformed.drop(columns=self.features)
        df_transformed=pd.concat([df_transformed,encoded_df],axis=1)
        logging.info("one-hot encoding completed")
        return df_transformed

class FeatureEngineering:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        logging.info("switching feature engineering strategy")
        self._strategy=strategy
    def apply_transformation(self,df:pd.DataFrame) ->pd.DataFrame:
        logging.info("Applying Feature ingineering strategy..")
        return self._strategy.apply_transformation(df)