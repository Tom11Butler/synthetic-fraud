"""
Defines a class to load the data
"""

import pandas as pd

class DataFetchStep():
    """
    Class for cleaning and preparing the data
    """
    
    
    def __init__(self, numerical_features, categorical_features):
        self._data = None
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features
        
        self._filepath = None
        
    @property
    def data(self):
        return self._data
    
    def fetch_data(self, filepath):
        """
        Returns the dataframe
        """
        self._filepath = filepath
        
        df = pd.read_csv(filepath)
        
        self._data = df
        
        return self._data