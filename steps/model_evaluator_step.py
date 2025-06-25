import logging
import mlflow
import pandas as pd 
from src.model_evaluation import ModelEvaluator,KnnEvaluationStrategy
from zenml import step
logging.basicConfig(level=logging.INFO,format="%(asctime)s -%(levelname)s - %(message)s")
@step
def model_evaluation_step(model,x_test:pd.DataFrame,y_test:pd.Series):
    if not isinstance(x_test,pd.DataFrame):
        raise TypeError("X_test must be pandas DataFrame.")
    if not isinstance(y_test,pd.Series):
        raise TypeError("y_test must be pandas Series.")
    logging.info("Aplying the sane process to the test data.")
    x_test_processed=model.named_steps["preprocessor"].transform(x_test)
    evaluator=ModelEvaluator(KnnEvaluationStrategy())
    evaluation_metrics=evaluator.evaluate(model.named_steps["model"],x_test_processed,y_test)
    if not isinstance(evaluation_metrics,dict):
        raise ValueError("Evaluation metrics must return dict form.")
    return evaluation_metrics