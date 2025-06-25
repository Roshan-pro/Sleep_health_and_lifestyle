from abc import ABC,abstractmethod
import pandas as pd 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

class MissingValueAnalysisTemplate(ABC):
    def analysis(self,df : pd.DataFrame):
        self.identify_missing_values(df)
        self.visualise_missing_values(df)
        pass
    @abstractmethod
    def identify_missing_values(self,df : pd.DataFrame):
        pass
        
    @abstractmethod
    def visualise_missing_values(self,df : pd.DataFrame):
        pass 
class SimpleMissingValueAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        missing_value=df.isna().sum()
        print(f"\nMissing value count by column :\n{missing_value > 0}")
    def visualise_missing_values(self,df : pd.DataFrame):
        print("Visualising Null values...")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull() ,cbar=False,cmap="viridis")
        plt.title("Missing values Heatmap")
        plt.show()