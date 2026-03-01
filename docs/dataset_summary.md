# Dataset Summary

To build the multimodal recommender system, BharatPropAdvisor leverages two distinct datasets: one primary tabular dataset for broader property features and one secondary image dataset for visual analysis. Both datasets are thoroughly integrated to perform deep learning-based multimodal fusion.

## 1. Primary Dataset: Pan-India Property Listings (2025)
**Source:** Provided via 4 split CSV files (`train_part1`, `train_part2`, `test_part1`, `test_part2`).

**Description:** A comprehensive tabular dataset representing realistic property listings across major Indian cities (such as Ahmedabad, Bengaluru, Chennai, Delhi NCR, Hyderabad, Kolkata, MMR, Pune). The split files have been combined into a unified dataset for training the Tabular Branch and Geospatial Clustering.

**Key Characteristics:**
- **Raw Size:** ~9,450 rows before cleaning, consisting of 25 distinct columns.
- **Cleaned Size:** 4,619 rows of highly valid data used for actual modeling.
- **Key Features Included:**
  - `City` and `Locality`
  - `PropertyType` (e.g., Apartment, Villa, Independent House, Studio, Penthouse, Row House)
  - `Price_INR` (converted to `Price_INR_Cr` for predictability)
  - `Facing` (crucial for Vastu compliance calculations)
  - `SuperBuiltUpArea_sqft`, `BuiltUpArea_sqft`, `CarpetArea_sqft`
  - `AmenitiesCount`, `Bathrooms`, `Balconies`, `Floor`, `TotalFloors`
  - `Latitude` / `Longitude` (used for geospatial KMeans clustering)

**Data Cleaning Procedures Applied:**
1. **Consolidation**: Merged all 4 split chunks into a master dataframe, appending a `Dataset_Split` column to flag original origin.
2. **Missing Value Imputation**: 
   - Filled missing categorical values (e.g., `Locality`, `Furnishing`, `Facing`) with the most frequent value (Mode).
   - Filled missing numerical values (e.g., `Bathrooms`, `Balconies`, `BuiltUpArea_sqft`) with the median to resist outliers.
3. **Outlier Filtering**: Validated data logic by dropping anomalies (e.g., removed properties with extreme `BuiltUpArea_sqft` over 15,000 sq ft or absurd pricing over ₹100 Cr).
4. **Target Mocking**: Since the core Kaggle dataset lacked multi-year ROI columns, realistic continuous target variables for `Actual_3Yr_ROI_Pct` and `Actual_5Yr_ROI_Pct` were synthetically generated to satisfy the Capstone modeling requirement.

**Usage:** This processed dataset (`data/processed/processed_pan_india_properties.csv`) forms the core backbone of the ROI prediction and ranked recommendation engine.

## 2. Secondary Dataset: Real Estate With Images
**Source:** [Kaggle - Real Estate with Image](https://www.kaggle.com/datasets/nikhilkushwaha2529/real-estate-with-image)

**Description:** A purely visual dataset containing thousands of interior and exterior photographs of real estate properties, categorized by room or perspective type.

**Key Characteristics:**
- **Size:** ~5,000 total real photos.
- **Categories:** 6 distinct folders (Bedroom, Kitchen, Bathroom, Living Room, Frontyard, Backyard).
- **Project Subset:** For our specific use case, we will sample and train on a refined subset of 1,500–2,000 high-quality images.

**Usage:** This dataset trains the Vision Transformer (ViT-base) branch of our architecture to learn visual cues not captured by tabular data (e.g., natural lighting, spaciousness, premium finishes). The output is a holistic "Visual Premium Score" and Grad-CAM heatmaps for explainability.
