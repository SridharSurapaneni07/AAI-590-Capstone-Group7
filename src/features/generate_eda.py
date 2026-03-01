import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def generate_eda(raw_dir, processed_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print(f"Loading raw datasets from {raw_dir}...")
    all_raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    df_raw_list = [pd.read_csv(f) for f in all_raw_files]
    if df_raw_list:
        df_raw = pd.concat(df_raw_list, ignore_index=True)
        
        # 1. Missing Values Heatmap (Before Cleaning)
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_raw.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Missing Values in Raw Dataset (Yellow = Missing)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cleaning_missing_values_heatmap.png'))
        print("Generated: cleaning_missing_values_heatmap.png")
        plt.close()

        # 2. Outliers Before Cleaning (BuiltUpArea_sqft)
        plt.figure(figsize=(10, 5))
        if 'BuiltUpArea_sqft' in df_raw.columns:
            import numpy as np
            data = df_raw['BuiltUpArea_sqft'].dropna()
            # Filter zero/negative values to allow log scaling
            data = data[data > 0]
            # Use manual log10 transform for better binning on extreme outliers
            sns.histplot(np.log10(data), bins=50, color='lightcoral')
            plt.title('Log10(BuiltUpArea_sqft) with Extreme Outliers [Raw]', fontsize=14)
            plt.xlabel('Log10(BuiltUp Area in sqft)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cleaning_outliers_before.png'))
            print("Generated: cleaning_outliers_before.png")
        plt.close()

    print(f"Loading processed data from {processed_file}...")
    df = pd.read_csv(processed_file)
    
    # 3. Outliers After Cleaning (BuiltUpArea_sqft)
    plt.figure(figsize=(10, 5))
    if 'BuiltUpArea_sqft' in df.columns:
        # Standard histogram after cleaning (no log scale needed)
        sns.histplot(df['BuiltUpArea_sqft'], bins=50, color='lightgreen', kde=True)
        plt.title('BuiltUpArea_sqft Distribution [Cleaned] (Linear Scale)', fontsize=14)
        plt.xlabel('BuiltUp Area (sqft)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cleaning_outliers_after.png'))
        print("Generated: cleaning_outliers_after.png")
    plt.close()

    # 4. Price Distribution by City
    plt.figure(figsize=(12, 6))
    if 'City' in df.columns and 'Price_INR_Cr' in df.columns:
        sns.boxplot(x='City', y='Price_INR_Cr', data=df, palette='Set2')
        plt.title('Property Price Distribution by City (INR Cr) [Post-Clean]', fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel('Price (Cr)')
        plt.ylim(0, df['Price_INR_Cr'].quantile(0.95)) 
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_price_distribution_by_city.png'))
        print("Generated: eda_price_distribution_by_city.png")
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation_heatmap.png'))
    print("Generated: eda_correlation_heatmap.png")
    plt.close()

    # 6. Property Type Count
    plt.figure(figsize=(10, 6))
    if 'PropertyType' in df.columns:
        sns.countplot(x='PropertyType', data=df, order=df['PropertyType'].value_counts().index, palette='viridis')
        plt.title('Count of Properties by Type', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_property_type_count.png'))
        print("Generated: eda_property_type_count.png")
    plt.close()

if __name__ == "__main__":
    raw_csv_dir = "data/raw"
    processed_csv = "data/processed/processed_pan_india_properties.csv"
    output_folder = "src/visualization"
    generate_eda(raw_csv_dir, processed_csv, output_folder)
