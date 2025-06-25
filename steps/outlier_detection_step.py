import logging
import pandas as pd 
from src.outlier_detection import OutlierDetector,ZcSoreOutlierDetection,IQROutlierDetection
from zenml import step
@step
def outlier_detection_step(df:pd.DataFrame,column_name:list[str])->pd.DataFrame:
    logging.info(f"Starting outlier deteection step with data.")
    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be pandas Data frame.")
    if not isinstance(df,pd.DataFrame):
        logging.error(f"Expected pandas DataFrame,got {type(df)}")
        raise ValueError("Input df must be pandas Data frame.")
    if not isinstance(column_name, list):
        column_name = [column_name]
    if not set(column_name).issubset(df.columns):
        logging.error(f"Invalid columns ({column_name}) name check!")
        raise ValueError(f"invalid column names :{column_name}")
    df_numeric=df.select_dtypes(include=[int,float])
    outlier_detector=OutlierDetector(IQROutlierDetection())
    outliers=outlier_detector.detect_outlier(df_numeric)
    df_cleaned=outlier_detector.handel_outliers(df_numeric,method="cap")
    return df_cleaned

# import logging
# import pandas as pd
# from src.outlier_detection import OutlierDetector, IQROutlierDetection
# from zenml import step

# @step
# def outlier_detection_step(df: pd.DataFrame, column_name: list[str]) -> pd.DataFrame:
#     logging.info("Starting outlier detection step with data.")

#     if df is None or not isinstance(df, pd.DataFrame):
#         raise ValueError("Input df must be a pandas DataFrame.")
#     if not isinstance(column_name, list):
#         column_name = [column_name]
#     if not set(column_name).issubset(df.columns):
#         raise ValueError(f"Invalid column names: {column_name}")

#     # Separate numeric and non-numeric data
#     df_numeric = df[column_name].select_dtypes(include=[int, float])
#     df_non_numeric = df.drop(columns=column_name)

#     # Detect and remove outliers
#     outlier_detector = OutlierDetector(IQROutlierDetection())
#     outliers = outlier_detector.detect_outlier(df_numeric)

#     # Keep rows without outliers (row-wise check)
#     mask = (~outliers).all(axis=1)
#     df_cleaned_numeric = df_numeric[mask]
#     df_cleaned_non_numeric = df_non_numeric[mask]

#     # Reset indices for proper alignment
#     df_cleaned_numeric = df_cleaned_numeric.reset_index(drop=True)
#     df_cleaned_non_numeric = df_cleaned_non_numeric.reset_index(drop=True)

#     # Combine
#     df_cleaned = pd.concat([df_cleaned_non_numeric, df_cleaned_numeric], axis=1)

#     logging.info(f"Shape after outlier removal: {df_cleaned.shape}")
#     logging.info(f"Columns after outlier removal: {df_cleaned.columns.tolist()}")

#     return df_cleaned

