import logging
import pandas as pd 
from sklearn.metrics import accuracy_score
from abc import ABC,abstractmethod
logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self,model,x_test:pd.DataFrame,y_test:pd.Series):
        pass
class KnnEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model, x_test:pd.DataFrame, y_test:pd.Series):
        logging.info("Predicting using the Train model.")
        y_pred=model.predict(x_test)
        logging.info("Calculating accuracy of model..")
        score=accuracy_score(y_test,y_pred)
        metrics={"TEST_SCORE":score}
        logging.info(f"Model Evaluation metrics :{metrics}")
        return metrics
class ModelEvaluator:
    def __init__(self,strategy:ModelEvaluationStrategy):
        self._strategy=strategy
    def set_strategy(self,strategy:ModelEvaluationStrategy):
        logging.info("Switching strategy..")
        self._strategy=strategy
    def evaluate(self,model,x_test:pd.DataFrame,y_test:pd.Series):
        return self._strategy.evaluate_model(model,x_test,y_test)