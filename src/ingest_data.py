import os
import zipfile
from abc import ABC,abstractmethod
import pandas as pd
class DataIngester(ABC):
    @abstractmethod
    def ingest(self,file_path : str) -> pd.DataFrame:
        pass
class UnzipData(DataIngester):
    def ingest(self, file_path : str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a zip file!")
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("Extracted_data")
            
        extracted_file=os.listdir("Extracted_data")
        csv_files=[f for f in extracted_file if f.endswith(".csv")]
        if len(csv_files)==0:
            raise ValueError("No csv file present!")
        if len(csv_files)>1:
            raise ValueError("Multiple csv is present choose 1!")
        csv_file = csv_files[0]
        csv_files_path = os.path.join("Extracted_data", csv_file)
        df = pd.read_csv(csv_files_path)
        return df
class Data:
    @staticmethod
    def get_data(file_extenion : str) -> DataIngester:
        if file_extenion ==".zip":
            return UnzipData()
        else:
            raise ValueError("check again")
if __name__=="__main__":
    path=r"C:\Users\rk186\OneDrive\Desktop\Sleep_health_and_lifestyle\data\archive (5).zip"
    extension=".zip"
    data_handler=Data.get_data(extension)
    data_handler.ingest(path)
    