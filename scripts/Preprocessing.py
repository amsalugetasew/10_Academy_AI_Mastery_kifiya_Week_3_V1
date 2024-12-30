import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
class Preprocessing:
    def __init__(self):
        self.df = {}
    def handling_missing_values(self, df):
        # Step 1: Count the occurrences of each category
        category_counts = df['Model'].value_counts()

        # Step 2: Calculate the mean of the category counts
        mean_count = category_counts.mean()

        # Step 3: Identify categories with counts >= mean
        categories_to_keep = category_counts[category_counts >= mean_count].index

        # Step 4: Filter the DataFrame to keep only rows with these categories
        df = df[df['Model'].isin(categories_to_keep)]
        
        # Step 1: Count the occurrences of each category
        category_counts = df['make'].value_counts()

        # Step 2: Calculate the first quartile (Q1) of the category counts
        q1_count = category_counts.quantile(0.25)

        # Step 3: Identify categories with counts > Q1
        categories_to_keep = category_counts[category_counts > q1_count].index

        # Step 4: Filter the DataFrame to keep only rows with these categories
        df = df[df['make'].isin(categories_to_keep)]
        # drop `NumberOfVehiclesInFleet` which have almost all values are missed
        df = df.drop(columns = ['NumberOfVehiclesInFleet','MaritalStatus', 'bodytype', 'Converted'])
        df = df.dropna(subset=['Bank', 'CustomValueEstimate'])
        # Define the columns to check for missing values
        columns_to_check = ['mmcode', 'VehicleType', 'make', 'Model', 'Cylinders', 'cubiccapacity', 'kilowatts', 'NumberOfDoors', 'VehicleIntroDate']
        # List of weak association features to drop
        weak_features = [
            "AlarmImmobiliser", "Citizenship", "Country", "CoverGroup", "CrossBorder",
            "ItemType", "Language", "NewVehicle", "Rebuilt", "StatutoryClass",
            "StatutoryRiskType", "TermFrequency", "WrittenOff", "Gender", "LegalType", "Section",
            "SubCrestaZone", "TrackingDevice"
        ]
        # Drop weak features from the DataFrame
        df = df.drop(columns=weak_features)
        # Drop rows where all values in these 10 columns are missing
        df = df.dropna(subset=columns_to_check, how='all')
        # For columns with missing numerical data, you can impute with mean
        numeric_cols = df.select_dtypes(include='number')
        category_cols = df.select_dtypes(include='object')
        for cols in numeric_cols:
            df[cols] = df[cols].fillna(df[cols].mean())
        for cols in category_cols:
            # For categorical columns, you can impute with the mode (most frequent value)
            df[cols] = df[cols].fillna(df[cols].mode()[0])
        return df
    def mean_count_resampling(self, df):
        # Step 1: Define categorical columns to process
        data_related_cols = ['TransactionMonth', 'VehicleIntroDate']
        category_cols = df.select_dtypes(include='object').columns.difference(data_related_cols)
        
        # Step 2: Resample for each categorical column
        for col in category_cols:
            # Calculate the mean count for the current column
            mean_count = int(round(df[col].value_counts().mean()))
            
            # Resample each category in the column
            df = (
                df.groupby(col, group_keys=False)  # Avoid adding a new index level
                .apply(lambda x: x.sample(n=mean_count, replace=len(x) < mean_count, random_state=42))
            )
            
        return df
    def feature_engineering(self, df):
        # Age of Vehicle (difference between transaction date and vehicle intro date)
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'])
        # Age of Vehicle: Based on VehicleIntroDate and TransactionMonth
        df['VehicleAge'] = (df['TransactionMonth'] - df['VehicleIntroDate']).dt.days / 365
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['year'] = df['TransactionMonth'].dt.year
        df['month'] = df['TransactionMonth'].dt.month
        df['day'] = df['TransactionMonth'].dt.day
        # Claim Frequency: Total claims over a specific period divided by the number of transactions.
        df['ClaimFrequency'] = df['TotalClaims'] / df['TransactionMonth'].dt.month
        df = df.drop(columns=['TransactionMonth', 'VehicleIntroDate'])  # Drop original date column if not needed
        return df
    def categorical_encoding(self, df):
        # One-Hot Encoding for features with more than two unique categories
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Label Encoding for binary features
        le = LabelEncoder()
        bool_cols = df.select_dtypes(include = 'bool').columns
        for cols in bool_cols:
            df[cols] = le.fit_transform(df[cols])
        
        return df

    def Train_Test_Split(self, df, test_size):
        # Define your target variable and features
        X = df.drop(columns=['TotalPremium', 'TotalClaims'])
        y = df['TotalClaims']  # or 'TotalPremium' based on your target variable

        # Split into 80% train and 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test 