from abc import ABC ,abstractmethod
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

class MultivariateAnalysisStrategy(ABC):
    def analyse(self,df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.geneate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self,df : pd.DataFrame):
        pass
    @abstractmethod
    def geneate_pairplot(self,df : pd.DataFrame):
        pass
class SimpleMultivariateAnalysis(MultivariateAnalysisStrategy):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(8,5))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor="blue")
        plt.title("Correlation of columns")
        plt.show()
    def geneate_pairplot(self,df:pd.DataFrame):
        plt.figure(figsize=(8,5))
        sns.pairplot(df)
        plt.suptitle("Pair plot of selected feature",y=1.02)
        plt.show()