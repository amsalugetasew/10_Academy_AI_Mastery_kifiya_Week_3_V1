import math
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
class EDA:
    def __init__(self):
        """Initializes the FileLoader with a list of file paths."""
        # self.file_paths = file_paths
        self.df = {}
    def hist_plot_numerical_cols(self, df):
        self.df = df
        numeric_cols = self.df.columns
        # Define the number of columns and calculate rows dynamically
        n_cols = 2  # Fixed number of columns
        n_rows = math.ceil(len(numeric_cols) / n_cols)  # Calculate rows dynamically

        # Set up subplots
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the 2D array of axes to a 1D array for easier iteration

        # Loop through numeric columns and plot histograms
        for i, column in enumerate(numeric_cols):
            df[column].hist(ax=axes[i], bins=20, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
            axes[i].tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()  # Adjust layout
        plt.show()  
    
    def bar_plot_4_categorical(self, df, category_cols):
    # Loop through categorical columns and plot bar charts
        for cols in category_cols:
            plt.figure(figsize=(8, 6))
            value_counts = df[cols].value_counts()
            colors = plt.cm.tab10(range(len(value_counts)))  # Generate distinct colors
            bars = None
            if len(value_counts.index) > 12:
                # Plot horizontal bar chart
                bars = plt.barh(value_counts.index, value_counts.values, color=colors)
                # Add labels to bars
                for bar in bars:
                    plt.annotate(f'{bar.get_width()}', 
                                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                xytext=(25, 0), textcoords="offset points",
                                va='center', ha='right', fontsize=8, color='black')
                plt.xticks([])
            else:
                # Plot vertical bar chart
                bars = plt.bar(value_counts.index, value_counts.values, color=colors)
                # Add labels to bars
                for bar in bars:
                    plt.annotate(f'{bar.get_height()}', 
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 5), textcoords="offset points",
                                va='bottom', ha='center', fontsize=10, color='black')
                plt.yticks([])

            # Add legend with different colors
            plt.legend(bars, value_counts.index, title=f"{cols} Categories", loc="upper right")
            if len(value_counts.index) > 12:
                # Set titles and labels
                plt.xlabel("Frequency")
                plt.ylabel(cols)
            else:
                # Set titles and labels
                plt.ylabel("Frequency")
                plt.xlabel(cols)
            plt.title(f"Distribution of {cols}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def correlation_analysis(self, df):
        # Ensure `TransactionMonth` is datetime and `PostalCode` is treated as a category
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['PostalCode'] = df['PostalCode'].astype('category')
        # Aggregate monthly data by PostalCode
        monthly_data = (
            df.groupby(['PostalCode', 'TransactionMonth'], observed=False)  # Explicitly set observed=False
            .agg({'TotalPremium': 'sum', 'TotalClaims': 'sum'})
            .reset_index()
            )
        # Calculate monthly changes
        monthly_data['PremiumChange'] = monthly_data.groupby('PostalCode', observed=False)['TotalPremium'].diff()
        monthly_data['ClaimsChange'] = monthly_data.groupby('PostalCode', observed=False)['TotalClaims'].diff()
        # Handle Missing value
        monthly_data.dropna(inplace=True)
        correlation_matrix = monthly_data[['PremiumChange', 'ClaimsChange']].corr()
        return df, monthly_data,correlation_matrix

    def correlation_heatmap(self, corr_matrix):
        # Heatmap for Correlation Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    def scatter_plot(self, monthly_data, sampled_postal_codes, col_1, col_2, col_3):
        for postal_code in sampled_postal_codes:
            subset = monthly_data[monthly_data[col_1] == postal_code]
            plt.figure(figsize=(8, 6))
            plt.scatter(subset[col_2], subset[col_3], alpha=0.7, label=f'PostalCode: {postal_code}')
            plt.title(f'Scatter Plot of Monthly Changes (PostalCode: {postal_code})')
            plt.xlabel(f'Change in {col_2}')
            plt.ylabel(f'Change in {col_3}')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.legend()
            plt.tight_layout()
            plt.show()
    def geo_comparision(self, df, cols):
        # Grouping by PostalCode or Province and calculating mean
        geo_trends = df.groupby([cols[0], cols[1]], observed=False).agg({
            cols[2]: 'count',  # Frequency of each cover type
            cols[3]: 'mean',  # Average premium per month
            cols[4]: 'nunique',  # Number of unique auto makes
            cols[5]: 'mean',  # Average claims
        }).reset_index()

        # Adding a percentage change for premium and claims
        geo_trends['PremiumChange'] = geo_trends.groupby(cols[0], observed=False)[cols[3]].pct_change()
        geo_trends['ClaimsChange'] = geo_trends.groupby(cols[0], observed=False)[cols[5]].pct_change()
        
        return geo_trends
    
    def geo_trend_analysis_visualization(self, geo_trends, n_length, cols):
        plt.figure(figsize=(12, 6))
        n_geo_trends = geo_trends.head(n_length)

        # Create the line plot
        sns.lineplot(
            data=geo_trends,
            x=cols[0],
            y=cols[1],
            hue=cols[2],
            legend='full',
            palette='tab10',  # Choose a bright, distinct color palette
        )

        # Enhance plot titles and labels
        plt.title('Trends in TotalPremium by Region', fontsize=14, weight='bold')
        plt.xlabel('Transaction Month', fontsize=12)
        plt.ylabel('Total Premium', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Customize the legend for better visibility
        legend = plt.legend(
            title='Region',
            title_fontsize=12,
            fontsize=10,
            loc='upper left',
            bbox_to_anchor=(1, 1),  # Position legend outside the plot
            frameon=True,           # Add a border to the legend
            shadow=True,            # Add a shadow for better visibility
        )

        # Set legend border and background color
        legend.get_frame().set_facecolor('lightgrey')
        legend.get_frame().set_edgecolor('black')

        plt.show()
    def box_plot_4_outlier_detection(self, df,numeric_columns):
        if 'NumberOfVehiclesInFleet' in df.columns:
            df.drop(columns=['NumberOfVehiclesInFleet'], inplace=True)
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))
        # Set up the plot grid
        plt.figure(figsize=(15, len(numeric_columns) * 3))

        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(len(numeric_columns), 1, i)
            sns.boxplot(data=df, x=column, color='skyblue')
            plt.title(f'Box Plot for {column}')
            plt.xlabel(column)
            plt.tight_layout()

        plt.show()
        
        
    def scatter_plot_for_corr(self, geo_trends):
        # Scatter plot for TotalPremium vs TotalClaims with valid PostalCodes
        fig = px.scatter(
            geo_trends,
            x='PremiumChange',
            y='ClaimsChange',
            color='Province',
            size='TotalPremium',  # Only positive values
            hover_data=['CoverType'],
            title='Premium vs Claims Change by Randomly Selected Province'
        )
        fig.show()
    
    def ferq_cover_type_by_province(self, df):
        auto_make_trends = df.groupby(['Province', 'CoverType'], observed=False).size().reset_index(name='Frequency')

        plt.figure(figsize=(12, 8))
        sns.barplot(data=auto_make_trends, x='Frequency', y='CoverType', hue='Province', dodge=False)
        plt.title('Frequency of Cover Type by Province')
        plt.xlabel('Frequency')
        plt.ylabel('Cover Type')
        plt.tight_layout()
        plt.show()
    def cover_type_dist_by_province(self,df):
        cover_trends = df.groupby(['Province', 'CoverType']).size().unstack(fill_value=0)
        plt.figure(figsize=(15, 10))
        sns.heatmap(cover_trends, cmap='RdYlGn', annot=True, fmt='d')
        plt.title('Heatmap of CoverType Distribution by Province')
        plt.xlabel('CoverType')
        plt.ylabel('Province')
        plt.tight_layout()
        plt.show()
    def trend_analysis_geo(self,geo_trends):
        # Aggregating data by Region for trend analysis

        premium_trends = geo_trends.groupby(["Province", "TransactionMonth"])["TotalPremium"].mean().reset_index()
        # Create a distinct color palette with enough unique colors
        palette = sns.color_palette("tab20", n_colors=16)  # "tab20" supports up to 20 unique colors

        # Visualization: Line Plot for Premium Trends Over Time
        plt.figure(figsize=(15, 10))
        sns.lineplot(
            data=premium_trends, 
            x="TransactionMonth", 
            y="TotalPremium", 
            hue="Province", 
            # marker="-", 
            palette=palette  # Apply the custom palette
        )
        plt.title("Premium Trends Over Time by Region")
        plt.ylabel("Average Premium")
        plt.xlabel("TransactionMonth")
        plt.legend(title="Region")
        plt.show()
        
    def make_preference_by_province(self, df):
        # Visualization 3: Auto Make Preferences by Region
        auto_make_dist = (
            df.groupby(["Province", "make"])["TotalPremium"]
            .count()
            .reset_index()
            .rename(columns={"TotalPremium": "Avg_Premium"})
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=auto_make_dist,
            x="Avg_Premium",  # Use 'Avg_Premium' on the x-axis
            y="Province",  # Use 'Province' on the y-axis
            hue="make"
        )
        plt.title("Auto Make Preferences by Region")
        plt.ylabel("Policy Count")
        plt.xlabel("Region")
        plt.legend(title="Auto Make")
        plt.show()
    def prem_by_cover_type_and_province(self, df):
        # Feature Engineering: Aggregate data by Region and Cover_Type
        aggregated = (
            df.groupby(["Province", "CoverType"])["TotalPremium"]
            .mean()
            .reset_index()
            .rename(columns={"TotalPremium": "Avg_Premium"})
        )

        # Visualization 1: Bar Chart of Average Premiums by Cover Type for Each Region
        plt.figure(figsize=(12, 16))
        sns.barplot(
            data=aggregated,
            x="Avg_Premium",  # Use 'Avg_Premium' on the x-axis
            y="Province",  # Use 'MainCrestaZone' on the y-axis
            hue="CoverType"
        )
        plt.title("Average Premiums by Cover Type and Region")
        plt.xlabel("Average Premium")
        plt.ylabel("Region")
        plt.legend(title="Cover Type", bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
        plt.tight_layout()  # Ensure proper layout
        plt.show()