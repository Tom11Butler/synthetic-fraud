"""
Defines the preprocessing class
"""

import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class PreProcessStep():
    """
    Object to apply to a dataframe and preprocess it
    """
    
    def __init__(self, numerical_columns, categorical_columns):
        self._numerical_columns = numerical_columns
        self._categorical_columns = categorical_columns
        self._encoded_categorical_columns = None
        self._pipeline = None
        
    @property
    def pipeline(self):
        return self._pipeline
    
    def _get_number_transformers(self, df):
        """
        Creates a list of tuples to map a column to its numerical transformer
        """
        
        num_transformers = None
        if self._numerical_columns is not None and len(self._numerical_columns)>0:
            
            #ensure all cols labelled as numeric are so
            for c in self._numerical_columns:
                df.loc[:,c] = pd.to_numeric(df.loc[:,c], errors='coerce')
                
            num_transformers = [
                (c, RobustScaler(), [c]) for c in self._numerical_columns
            ]
            
        return num_transformers
    
    def _get_categorical_transformers(self, df):
        """
        Creates a list of tuples to map a column to its categorical transformer
        """
        
        cat_transformers = None
        if self._categorical_columns is not None and len(self._categorical_columns)>0:
            cat_transformers = [
                (c, OneHotEncoder(), [c]) for c in self._categorical_columns
            ]
        
        return cat_transformers
    
    def _build_pipeline(self, df):
        """
        Creates the pipeline to transform the data
        """
        number_transformers = self._get_number_transformers(df)
        categorical_transformers = self._get_categorical_transformers(df)
        
        pipeline = ColumnTransformer(
            number_transformers
            + categorical_transformers
        )
        
        self._pipeline = pipeline
        
    def fit_transform(self, df, allow_refit=False):
        if self._pipeline is not None and not allow_refit:
            raise Exception("Refitting is not allowed by default")
            
        self._build_pipeline(df)
        self._pipeline.fit(df)
        self._update_encoded_categories()
        
        return self._pipeline.transform(df)
    
    def transform(self, df):
        return self._pipeline.transform(df)
    
    def _update_encoded_categories(self):
        """
        Update the conded categories data so a dataframe with the appropriate
        column names may be created
        """
        encoded_categorical_columns = []
        
        encoders = [t[:2] for t in self._pipeline.transformers_ if
                    t[0] in self._categorical_columns]
        
        for column_name, enc in encoders:
            encoded_categorical_columns = encoded_categorical_columns + [
                f'{column_name}_{cat}' for cat in enc.categories_[0]
            ]
            
        self._encoded_categorical_columns = encoded_categorical_columns