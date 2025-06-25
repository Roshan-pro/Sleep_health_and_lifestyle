import pandas as pd 
from src.handle_missing_values import MissingValueHandeler,FillMissingValueStrategy

from zenml import step

@step
def handel_missing_values_step(df : pd.DataFrame,strategy : str ="mean") ->pd.DataFrame:
    if strategy in ["mean","median","mode","constant"]:
        handeler=MissingValueHandeler(FillMissingValueStrategy(method=strategy))
    else:
        raise ValueError(f"such {strategy} does not exist")
    cleaned_df=handeler.handel_missing_values(df)
    return cleaned_df