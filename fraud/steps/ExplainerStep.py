"""
Defines the explainer class

TODO: fill in class method definitions
"""

class ExplainerStep():
    """
    Object to explain model predictions
    """
    
    def __init__(self, explainer, transformers):
        self._explainer = explainer
        self._transformers = transformers
        
        
    def _inverse_transform(self, col, val):
        pass
    
    def _report_most_important_features(self, instance):
        pass
    
    def _shape_force_plot_instance(self, instance):
        pass
    
    def report_tp_fp_tn_fs_explanations(self, predictions, X):
        pass
    
    
    def all_shap_values(self, X):
        pass
        
    