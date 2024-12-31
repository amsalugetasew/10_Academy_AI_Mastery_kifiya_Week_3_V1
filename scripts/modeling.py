
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_error, r2_score
class Modeling:
    def __init__(self):
        self.df = {}
    
    # Linear Regression
    def linear_regression(self, X_train, y_train):
        # Train the model
        lin_reg = LinearRegression()
        model = lin_reg.fit(X_train, y_train)
        return model
    # Decession Tree
    def decision_tree(self, X_train, y_train):
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model
    # Random Forest Tree
    def random_forest(self, X_train, y_train):
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        return rf_reg
    
    # XGBost  
    def XGBRegressor_model(self,X_train, y_train):
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_reg.fit(X_train, y_train)
        return xgb_reg

    # Model performance 
    def model_performamnce(self, model, X_test, y_test):
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
        print(f"Root Mean Squared Error: {root_mean_squared_error(y_test, y_pred)}")
        print(f'R-squared: {r2_score(y_test, y_pred)}')
        return y_pred
    
    def model_comparison(self,y_test, y_pred, y_pred_dt, y_pred_rf, y_pred_xgb):
        # Model comparison 
        model_comparison = {
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
            'MSE': [mean_squared_error(y_test, y_pred), mean_squared_error(y_test, y_pred_dt), mean_squared_error(y_test, y_pred_rf), mean_squared_error(y_test, y_pred_xgb)],
            'R-squared': [r2_score(y_test, y_pred), r2_score(y_test, y_pred_dt), r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_xgb)],
            'MAE': [mean_absolute_error(y_test, y_pred), mean_absolute_error(y_test, y_pred_dt), mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_xgb)]
        }

        model_comparison_df = pd.DataFrame(model_comparison)
        model_comparison_df
        return model_comparison_df
        
        
    