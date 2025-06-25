import logging
from abc import ABC,abstractmethod
from typing import Any
import pandas as pd 
from sklearn.metrics  import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,x_train:pd.DataFrame,y_train:pd.Series):
        pass
class KNeighborsClassifierStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, x_train:pd.DataFrame, y_train:pd.Series):
        if not isinstance(x_train,pd.DataFrame):
            raise ValueError("X_train must be Pandas Data Frame.")
        if not isinstance(y_train,pd.Series):
            raise ValueError("y_train must be Pandas Data Series.")
        logging.info("Initialization KNeighborsClassifier model with scaling.")
        pipeline=Pipeline([
            ("Scaler",StandardScaler()),
            ("model",KNeighborsClassifier())
        ])
        logging.info("Training KNeighborsClassifier model..")
        pipeline.fit(x_train,y_train)
        logging.info("Model training completed.")
        return pipeline
    
class ModelBuilder:
    def __init__(self,strategy:ModelBuildingStrategy):
        self._strategy=strategy
    def set_strategy(self,strategy:ModelBuildingStrategy):
        logging.info("Switching model building strategy.")
        self._strategy=strategy
    def build_model(self,x_train:pd.DataFrame,y_train:pd.Series):
        logging.info("Building and training the model using KNN..")
        return self._strategy.build_and_train_model(x_train,y_train)