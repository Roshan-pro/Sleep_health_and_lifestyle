from abc import ABC,abstractmethod
import pandas as pd 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

class UnivariableAnalysisStrategy(ABC):
    @abstractmethod
    def analysis(self,df : pd.DataFrame,feature: str):
        pass
class NumericalUnivariableAnalysis(UnivariableAnalysisStrategy):
    def analysis(self, df: pd.DataFrame, feature:str):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature],kde=True,bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
class CategoricalUnivariableAnalysis(UnivariableAnalysisStrategy):
    def analysis(self, df: pd.DataFrame, feature:str):
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature,data=df,palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
class UnivariableAnalysis:
    def __init__(self,strategy : UnivariableAnalysisStrategy):
        self._strategy=strategy
    def set_strategy(self,strategy: UnivariableAnalysisStrategy):
        self._strategy=strategy
    def execute_analysis(self,df:pd.DataFrame,feature:str):
        self._strategy.analysis(df,feature)
