import pandas as pd
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, chi2
from scipy import stats
class HypothessTest:
    def __init__(self):
        self.df = {}
    
    # Step 1: Calculte Margine 
    def calculate_profit_margin(self, data, feature1, feature2, ):
        """Calculate profit margin and add it as a new column."""
        self.df = data
        self.df['ProfitMargin'] = self.df[feature1] - self.df[feature2]
        return self.df
    # Step 2: Perform ANOVA Test
    def perform_anova(self, data, group_column, target_column):
        """Perform ANOVA test for a numerical target across multiple groups."""
        self.df = data
        groups = self.df[group_column].unique()
        grouped_data = [self.df[self.df[group_column] == group][target_column].dropna() for group in groups]
        anova_stat, p_value = f_oneway(*grouped_data)
        return anova_stat, p_value
    # Step 3: Perform T-Test
    def perform_t_test(self, data, group_column, target_column, group_a, group_b):
        """Perform T-Test between two groups."""
        group_a_data = data[data[group_column] == group_a][target_column].dropna()
        group_b_data = data[data[group_column] == group_b][target_column].dropna()
        t_stat, p_value = ttest_ind(group_a_data, group_b_data, nan_policy='omit')
        return t_stat, p_value
    
    # Step 4: Perform Chi-Squared Test
    def perform_chi2_test(self,data, group_column, target_column):
        """Perform Chi-Squared test for categorical data."""
        contingency_table = pd.crosstab(data[group_column], data[target_column])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2_stat, p_value
    # Step 5: Interpret Results
    def interpret_results(self, p_value, alpha=0.05):
        """Interpret the results based on the p-value and significance level."""
        if p_value < alpha:
            return "Reject the null hypothesis: Significant differences found."
        else:
            return "Fail to reject the null hypothesis: No significant differences found."
    # Chi-squared test for categorical vs  target attributes on a data
    def chi_squared_test_with_target_var(self, df, target, categorical_columns, alpha=0.05):
        results = []
        for col in categorical_columns:
            # Creating a contingency table
            contingency_table = pd.crosstab(df[col], df[target])
            
            # Applying Chi-Squared test
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Compute the Chi-Squared threshold based on DOF and significance level
            chi2_threshold = chi2.ppf(1 - alpha, dof)
            
            # Interpretation of p-value
            if p_val < alpha:
                p_interpretation = "Significant (Reject H0)"
            else:
                p_interpretation = "Not Significant (Accept H0)"
            
            # Interpretation of Chi-Squared statistic based on threshold
            if chi2_stat > chi2_threshold:
                chi2_interpretation = "Strong association"
            elif (chi2_stat > 0.80 * chi2_threshold) & (p_val > 0.5):  # Adjust weak-to-moderate range
                chi2_interpretation = "Moderate association"
            else:
                chi2_interpretation = "Weak or no association"
            
            # Append results to the list
            results.append({
                'Feature': col,
                'Chi2': chi2_stat,
                'Chi2 Threshold': chi2_threshold,
                'P-Value': p_val,
                'Chi2 Interpretation': chi2_interpretation,
                'P-Value Interpretation': p_interpretation
            })

        # Convert results to a DataFrame
        return pd.DataFrame(results)
    # Chi-squared test for categorical vs  categorical attributes on a data
    def chi_squared_test(self, df, categorical_columns, alpha=0.05):
        results = []
        # Iterate over all pairs of categorical features
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:  # Pairs of features
                # Creating a contingency table between two categorical features
                contingency_table = pd.crosstab(df[col1], df[col2])
                
                # Skip empty tables or tables that are too small to apply Chi-squared
                if contingency_table.empty or contingency_table.shape[0] == 1 or contingency_table.shape[1] == 1:
                    print(f"Skipping {col1} vs {col2} due to empty or invalid contingency table.")
                    continue

                # Applying Chi-Squared test
                chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                
                p_interpretation = None
                # Interpretation based on p-value
                if p_val <= alpha:
                    p_interpretation = "Significant differences found (Reject H0)"
                else:
                    p_interpretation = "Not Significant differences found (Accept H0)"
                
                # General interpretation of Chi-squared statistic
                if chi2_stat > 10:
                    chi2_interpretation = "Strong association"
                elif chi2_stat > 5:
                    chi2_interpretation = "Moderate association"
                else:
                    chi2_interpretation = "Weak association"
                
                # Append results to the list with feature names
                results.append({
                    'cat_feature1': col1,
                    'cat_feature2': col2,
                    'Chi2': chi2_stat,
                    'P-Value': p_val,
                    'Chi2 Interpretation': chi2_interpretation,
                    'P-Value Interpretation': p_interpretation
                })
        
        # Convert results to a DataFrame
        return pd.DataFrame(results)

    # T-test for numerical data (assuming target is binary)
    def t_test(self, df, target, numerical_columns):
        results = {}
        for col in numerical_columns:
            group1 = df[df[target] == 0][col]
            group2 = df[df[target] == 1][col]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            results[col] = {'T-Statistic': t_stat, 'P-Value': p_val}
        return results

        