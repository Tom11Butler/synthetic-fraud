"""
Master file for the features dictionary
"""

FEATURES = dict()

FEATURES['label_column'] = 'isFraud'
# FEATURES['id_column'] = ''

FEATURES['numerical_columns'] = [
    'amount'
    , 'oldbalanceOrg'
    , 'newbalanceOrig'
    , 'oldbalanceDest'
    , 'newbalanceDest'
]

FEATURES['categorical_columns'] = [
    'type'
]