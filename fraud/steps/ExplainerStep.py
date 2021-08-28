"""
Defines the explainer class

TODO: fill in class method definitions
"""

import shap

class ExplainerStep():
    """
    Object to explain model predictions
    """
    
    def __init__(self, explainer, transformers, feature_columns):
        self._explainer = explainer
        self._transformers = transformers
        self._feature_columns = feature_columns
        
        self._predictions = None
        
        
    def _inverse_transform(self, col, val):
        return self._transformers[col]\
                .inverse_transform(val.reshape(-1,1))[0][0]
    
    def _report_most_important_features(self, instance):
        """
        Reports the top five most important features for a given local
        prediction.
        """
        
        shap_values = self._explainer.shap_values(instance)
        
        df_shap = pd.DataFrame(
            shap_values[1].reshape(-1,1)
            , index=cols
            , columns=['shap_value']
        )
        df_shap['abs_shap_value'] = abs(df_shap['shap_value'])
        df_shap.sort_values('abs_shap_value', ascending=False, inplace=True)
        
        
    
    def _shap_force_plot_instance(self, instance):
        display(shap.force_plot(
            self._explainer.expected_value[1]
            , self._explainer.shap_values(instance)[1]
            , instance
            , self._feature_columns))
    
    def report_tp_fp_tn_fs_explanations(self, predictions, X):
        self._predictions = predictions
        
        tp = self._predictions.query("true==pred").query("true==1").index[0]
        fp = self._predictions.query("true!=pred").query("true==1").index[0]
        tn = self._predictions.query("true==pred").query("true==0").index[0]
        fn = self._predictions.query("true!=pred").query("true==0").index[0]
        
        for pred_type, index in zip(['tp','fp','tn','fn'], [tp,fp,tn,fn]):
            print(f"Example explanation for a {pred_type} prediction ...")
            print("----------------")
            
            instance = X[index]
            
            # self._report_most_important_features(instance)
            self._shap_force_plot_instance(instance)
    
    def all_shap_values(self, X):
        pass
        
    