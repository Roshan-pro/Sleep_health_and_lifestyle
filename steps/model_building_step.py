import logging
from typing import Annotated

import mlflow
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from zenml import ArtifactConfig,step
from zenml.client import Client
from zenml import Model
experiment_tracker = Client().active_stack.experiment_tracker


model =Model(
    name="Predicting_Sleep_Disorder",
    version=None,
    license="Apache 2.0",
    description="Predicting Sleep Disorder"
)
# @step(enable_cache=False,experiment_tracker=experiment_tracker.name,model=model)
# def model_building_step(x_train:pd.DataFrame,y_train:pd.Series):
#     if not isinstance(x_train,pd.DataFrame):
#             raise ValueError("X_train must be Pandas Data Frame.")
#     if not isinstance(y_train,pd.Series):
#         raise ValueError("y_train must be Pandas Data Series.")
#     categorical_cols=x_train.select_dtypes(include=["object"]).columns
#     numerical_cols=x_train.select_dtypes(include=["int","float"]).columns
#     logging.info(f"categorical_cols :{categorical_cols.tolist()}")
#     logging.info(f"numerical_cols :{numerical_cols.tolist()}.")
    
#     numerical_transformer=SimpleImputer(strategy="mean")
#     categorical_transformer=Pipeline([
#         ("imputer",SimpleImputer(strategy="most_frequent")),
#         ("onehot",OneHotEncoder(handle_unknown="ignore"))
#     ])
#     preprocessor=ColumnTransformer([
#         ("num",numerical_transformer,numerical_cols),
#         ("cat",categorical_transformer,categorical_cols)
#     ])
#     logging.info("Initialization KNeighborsClassifier model with scaling.")
#     pipeline=Pipeline([
#         ("preprocessor",preprocessor),
#         ("model",KNeighborsClassifier())
#     ])
#     if not mlflow.active_run():
#         mlflow.start_run()
#     try:
#         mlflow.sklearn.autolog()
#         logging.info("Training KNeighborsClassifier model..")
#         pipeline.fit(x_train,y_train)
#         logging.info("Model training completed.")
#         one_encoder=(
#             # pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
#             pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
#         )
#         one_encoder.fit(x_train[categorical_cols])
#         expected_columns=numerical_cols.tolist() +list(
#             one_encoder.get_feature_names_out(categorical_cols)
#         )
#         logging.info(f"Model expect the folling columns :{expected_columns}")
#     except Exception as e:
#         logging.error(f"Error during model training : {e}")
#         raise e
#     finally:
#         mlflow.end_run()
#     return pipeline
@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(x_train: pd.DataFrame, y_train: pd.Series):
    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("X_train must be Pandas Data Frame.")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be Pandas Data Series.")

    categorical_cols = x_train.select_dtypes(include=["object"]).columns
    numerical_cols = x_train.select_dtypes(include=["int", "float"]).columns

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier())
    ])

    try:
        mlflow.sklearn.autolog()  # ✅ No start_run() or end_run()
        pipeline.fit(x_train, y_train)
        one_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        one_encoder.fit(x_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(one_encoder.get_feature_names_out(categorical_cols))
        logging.info(f"Model expects the following columns: {expected_columns}")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    return pipeline
