import os
import pandas as pd

class FileLoader:
    def __init__(self):
        """Initializes the FileLoader with a list of file paths."""
        # self.file_paths = file_paths
        self.dataframes = {}
    def lead_text_files(self):
        self.dataframes = pd.read_csv("../src/Data/first/MachineLearningRating_v3/MachineLearningRating_v3.txt", delimiter='|', low_memory=False)
        df = pd.read_csv("../src/Data/second/MachineLearningRating_v3/MachineLearningRating_v3.txt", delimiter='|', low_memory=False)
        return self.dataframes, df
    def save_csv_file(self, df):
        self.dataframes = df.to_csv("../src/Data/first/MachineLearningRating_v3/MachineLearningRating_v3.csv", index=False)
        return self.dataframes 
    def read_csv_file(self):
        self.dataframes = pd.read_csv("../src/Data/first/MachineLearningRating_v3/MachineLearningRating_v3.csv", low_memory=False)
        return self.dataframes
   