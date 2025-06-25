from abc import ABC,abstractmethod
import pandas as pd 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

class BivariableAnalysisStrategy(ABC):
    @abstractmethod
    def analyse(self,df:pd.DataFrame,feature1:str,feature2:str):
        pass
class NumericalBivariableAnalysis(BivariableAnalysisStrategy):
    def analyse(self, df:pd.DataFrame, feature1:str, feature2:str):
        plt.figure(figsize=(8,5))
        plt.title(f"{feature1} VS {feature2}")
        sns.scatterplot(df,x=feature1,y=feature2)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
class CategoricalBivariableAnalysis(BivariableAnalysisStrategy):
    def analyse(self, df:pd.DataFrame, feature1:str, feature2:str):
        plt.figure(figsize=(8,5))
        plt.title(f"{feature1} VS {feature2}")
        sns.boxplot(df,x=feature1,y=feature2)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()
class BivariableAnalysis:
    def __init__(self,strategy : BivariableAnalysisStrategy):
        self._strategy=strategy
    def set_strategy(self,strategy:BivariableAnalysisStrategy):
        self._strategy=strategy
    def execute_analysis(self,df:pd.DataFrame,feature1:str,feature2:str):
        self._strategy.analyse(df,feature1,feature2)