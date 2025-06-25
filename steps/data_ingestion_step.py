import pandas as pd 
from src.ingest_data import Data
from zenml import step
@step
def data_ingestion_step(file_path: str)->pd.DataFrame:
    file_extention=".zip"
    data_ingester=Data.get_data(file_extention)
    df=data_ingester.ingest(file_path)
    return df