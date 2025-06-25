from abc import ABC,abstractmethod
import pandas as pd 

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self,df : pd.DataFrame):
        pass
class AboutData(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print(f"\nData types and Non-values:\n{df.info()}")
class SummaryStatisticInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df : pd.DataFrame):
        print(f"\nSummary Statistics (Numerical features):\n{df.describe()}")
        print(f"\nSummary Statistic (Categorical features):\n{df.describe(include=["object"])}")
        
class DataInspector:
    def __init__(self,strategy : DataInspectionStrategy):
        self._strategy=strategy
    def execute_inspection(self,df : pd.DataFrame):
        self._strategy.inspect(df)
        