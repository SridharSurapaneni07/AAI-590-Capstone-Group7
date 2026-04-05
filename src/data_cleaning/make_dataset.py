import pandas as pd
import numpy as np
import os
import glob

def clean_tabular_data(input_dir, output_dir):
    """
    Reads the part files (train1, train2, test1, test2), concatenates them,
    cleans missing values, handles outliers, and saves as a single processed dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all CSV parts
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return
        
    print(f"Found {len(all_files)} files: {all_files}")
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        # Identify if this is from train or test set
        df['Dataset_Split'] = 'Train' if 'train' in os.path.basename(file).lower() else 'Test'
        df_list.append(df)
        
    # Concatenate all parts
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"Master dataset shape before cleaning: {master_df.shape}")
    
    # 1. Clean Missing Values
    # Categorical: Fill with Mode
    for col in ['Locality', 'Furnishing', 'BuildingType', 'Facing']:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(master_df[col].mode()[0])
            
    # Numerical: Fill with Median
    for col in ['Bathrooms', 'Balconies', 'SuperBuiltUpArea_sqft', 'BuiltUpArea_sqft', 'CarpetArea_sqft', 'YearBuilt', 'AgeYears', 'AmenitiesCount']:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
            master_df[col] = master_df[col].fillna(master_df[col].median())
            
    # Location coordinates: Drop rows missing both Lat/Long
    if 'Latitude' in master_df.columns and 'Longitude' in master_df.columns:
        master_df = master_df.dropna(subset=['Latitude', 'Longitude'])
        
    # Target Variable: Price_INR
    if 'Price_INR' in master_df.columns:
        master_df = master_df.dropna(subset=['Price_INR'])
        master_df['Price_INR_Cr'] = master_df['Price_INR'] / 10000000.0 # Convert to Crores
        
    # 2. Filter obvious outliers
    if 'BuiltUpArea_sqft' in master_df.columns:
        # Keep properties sensible in area (e.g. 200 sqft to 10,000 sqft)
        master_df = master_df[(master_df['BuiltUpArea_sqft'] > 200) & (master_df['BuiltUpArea_sqft'] < 15000)]
        
    if 'Price_INR_Cr' in master_df.columns:
         # Keep prices under boundary (e.g. 100 Cr max)
         master_df = master_df[master_df['Price_INR_Cr'] <= 100.0]

    # Create Mock Target Columns for ROI (since Kaggle data only has Price)
    # The capstone requires predicting 3-year and 5-year ROI.
    np.random.seed(42)
    master_df['Actual_3Yr_ROI_Pct'] = np.random.uniform(5.0, 35.0, len(master_df)).round(2)
    master_df['Actual_5Yr_ROI_Pct'] = (master_df['Actual_3Yr_ROI_Pct'] * np.random.uniform(1.2, 1.8, len(master_df))).round(2)
    
    print(f"Master dataset shape after cleaning: {master_df.shape}")
    
    # Save processed dataframe
    output_file = os.path.join(output_dir, "processed_pan_india_properties.csv")
    master_df.to_csv(output_file, index=False)
    print(f"Saved processed dataset to {output_file}")

if __name__ == "__main__":
    clean_tabular_data("data/raw", "data/processed")
