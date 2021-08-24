import datetime

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (plot_confusion_matrix
                             , confusion_matrix
                             , classification_report)

from fraud.config.config import PARAMS
from fraud.config.features import FEATURES

from fraud.steps.DataFetchStep import DataFetchStep
from fraud.steps.PreProcessStep import PreProcessStep


if __name__=="__main__":
    time = datetime.datetime.now()
    PARAMS['model_id'] = time.strftime("%Y%m%d%H%M%S")
    
    data_fetch_step = DataFetchStep(FEATURES['numerical_columns']
                                    , FEATURES['categorical_columns'])
    
    df = data_fetch_step.fetch_data(PARAMS['input_data'])
    
    preprocess_step = PreProcessStep(
        FEATURES['numerical_columns']
        , FEATURES['categorical_columns']
    )
    
    X = df[FEATURES['numerical_columns']+FEATURES['categorical_columns']]
    y = df[FEATURES['label_column']]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    X_train = preprocess_step.fit_and_transform(X_train)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    dump(clf, f"{PARAMS['output_dir']}/{PARAMS['model_id']}_model.joblib")