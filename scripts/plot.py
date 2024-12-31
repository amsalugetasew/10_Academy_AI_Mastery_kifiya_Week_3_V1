import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import lime
from lime.lime_tabular import LimeTabularExplainer
class Plot:
    def __init__(self):
        self.df = {}
    
    def feature_importance(self, model, X_train, title):
            # Random Forest feature importance
            feat_importance_rf = model.feature_importances_
            feat_names = X_train.columns
            plt.barh(feat_names[:30], feat_importance_rf[:30])
            plt.xlabel('Feature Importance')
            plt.title(title)
            plt.show()

            # XGBoost feature importance
            xgb.plot_importance(model, importance_type='weight')
            plt.show() 
    # Function to plot feature importance for each model
    def plot_feature_importance(self, models, X_train):
        # Set up the figure with one column and as many rows as there are models
        plt.figure(figsize=(15, 20))
        
        # Loop through each model
        for idx, (model_name, model) in enumerate(models, start=1):
            if hasattr(model, 'coef_'):  # Linear Regression has 'coef_' for feature importance
                importance = np.abs(model.coef_)
            elif hasattr(model, 'feature_importances_'):  # Decision Tree, Random Forest, and XGBoost
                importance = model.feature_importances_
            else:
                print(f"{model_name} does not have feature importance")
                continue
            
            # Sort feature importance and get top 30
            sorted_idx = np.argsort(importance)[1:-1]
            top_idx = sorted_idx[:30]
            top_features = X_train.columns[top_idx]
            top_importance = importance[top_idx]
            
            # Plot
            plt.subplot(len(models), 1, idx)
            plt.barh(top_features, top_importance)
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} Feature Importance')
            plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
        
        plt.tight_layout()
        plt.show()
        
    
    def plot_lime_explanation(self, model, X_train, X_test, instance_idx):
        # Check if the model is for classification or regression
        if hasattr(model, 'predict_proba'):  # Classification model
            mode = 'classification'
            predict_fn = model.predict_proba
            class_names = [str(i) for i in range(model.n_classes_)]
        else:  # Regression model (Linear Regression, Decision Tree, Random Forest, XGBoost)
            mode = 'regression'
            predict_fn = model.predict
            class_names = None  # No class names for regression
        
        # Initialize the LimeTabularExplainer
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            mode=mode,  # Choose mode based on the model type
            class_names=class_names  # Only needed for classification
        )
        
        # Explain a single instance's prediction
        exp = explainer.explain_instance(X_test.iloc[instance_idx].values, predict_fn)
        
        # Plot the explanation (you can also use exp.show_in_notebook() in Jupyter)
        exp.as_pyplot_figure()