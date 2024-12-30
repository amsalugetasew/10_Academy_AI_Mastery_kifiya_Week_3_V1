import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, root_mean_squared_log_error, root_mean_squared_error, r2_score

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
    
    def feature_importance(self, model, X_train, title):
        # Random Forest feature importance
        feat_importance_rf = model.feature_importances_
        feat_names = X_train.columns
        plt.barh(feat_names, feat_importance_rf)
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.show()

        # XGBoost feature importance
        xgb.plot_importance(model, importance_type='weight')
        plt.show() 
        