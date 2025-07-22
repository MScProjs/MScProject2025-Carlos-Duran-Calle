# encoding_utils.py
# Improved Categorical Variable Encoding Strategy
# Tailored encoding for different variable types

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import os

def encode_categorical_variables(X_train, X_test=None):
    """
    Apply appropriate encoding for each categorical variable type:
    - region: One-hot encoding (nominal)
    - highest_education: Ordinal encoding (ordinal)
    - imd_band: Ordinal encoding with proper NaN handling (ordinal)
    - age_band: Ordinal encoding (ordinal)
    - disability: Binary encoding (binary)
    """
    
    # Create copies to avoid modifying original data
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None
    
    print("🔧 APPLYING TAILORED ENCODING STRATEGIES")
    print("=" * 50)
    
    # Store encoders for later use on test data
    encoders = {}
    
    # ================================================================
    # 1. REGION - One-Hot Encoding (Nominal Variable)
    # ================================================================
    print("🌍 Processing 'region' (Nominal - One-Hot Encoding)")
    
    # One-hot encode region
    region_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    
    # Fit on training data
    region_encoded_train = region_encoder.fit_transform(X_train_encoded[['region']])
    region_feature_names = [f'region_{cat}' for cat in region_encoder.categories_[0][1:]]  # Skip first due to drop='first'
    
    # Create DataFrame for encoded features
    region_df_train = pd.DataFrame(region_encoded_train, 
                                   columns=region_feature_names, 
                                   index=X_train_encoded.index)
    
    # Apply to test data if provided
    if X_test_encoded is not None:
        region_encoded_test = region_encoder.transform(X_test_encoded[['region']])
        region_df_test = pd.DataFrame(region_encoded_test, 
                                      columns=region_feature_names, 
                                      index=X_test_encoded.index)
    
    print(f"   → Created {len(region_feature_names)} binary features")
    print(f"   → Original categories: {list(region_encoder.categories_[0])}")
    
    encoders['region'] = region_encoder
    
    # ================================================================
    # 2. HIGHEST_EDUCATION - Ordinal Encoding (Ordinal Variable)
    # ================================================================
    print("\n🎓 Processing 'highest_education' (Ordinal)")
    
    # Define education hierarchy (adjust based on your data)
    education_order = [
        'No Formal quals',
        'Lower Than A Level', 
        'A Level or Equivalent',
        'HE Qualification',
        'Post Graduate Qualification'
    ]
    
    # Check what education levels exist in your data
    unique_education = X_train_encoded['highest_education'].unique()
    print(f"   → Found education levels: {sorted(unique_education)}")
    
    # Create ordinal encoder
    education_encoder = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit and transform training data
    X_train_encoded['highest_education_ord'] = education_encoder.fit_transform(X_train_encoded[['highest_education']]).flatten()
    
    # Apply to test data if provided
    if X_test_encoded is not None:
        X_test_encoded['highest_education_ord'] = education_encoder.transform(X_test_encoded[['highest_education']]).flatten()
    
    print(f"   → Encoded as: {dict(zip(education_order, range(len(education_order))))}")
    
    encoders['highest_education'] = education_encoder
    
    # ================================================================
    # 3. IMD_BAND - Ordinal Encoding with NaN Handling (Ordinal Variable)
    # ================================================================
    print("\n📊 Processing 'imd_band' (Ordinal with Missing Values)")
    
    # Handle missing values in IMD band
    missing_count = X_train_encoded['imd_band'].isnull().sum()
    print(f"   → Found {missing_count} missing values in training data")
    
    # Define IMD band order (typically 0-100% in bands)
    # Adjust these based on your actual IMD band categories
    imd_order = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                 '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    # Check actual categories in your data
    unique_imd = X_train_encoded['imd_band'].dropna().unique()
    print(f"   → Found IMD bands: {sorted(unique_imd)}")
    
    # Strategy: Treat missing as separate category or impute with median
    # Option 1: Create 'Unknown' category for missing values
    X_train_encoded['imd_band_filled'] = X_train_encoded['imd_band'].fillna('Unknown')
    if X_test_encoded is not None:
        X_test_encoded['imd_band_filled'] = X_test_encoded['imd_band'].fillna('Unknown')
    
    # Add 'Unknown' to categories if it exists
    imd_order_with_unknown = imd_order + ['Unknown'] if missing_count > 0 else imd_order
    
    imd_encoder = OrdinalEncoder(categories=[imd_order_with_unknown], handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded['imd_band_ord'] = imd_encoder.fit_transform(X_train_encoded[['imd_band_filled']]).flatten()
    
    if X_test_encoded is not None:
        X_test_encoded['imd_band_ord'] = imd_encoder.transform(X_test_encoded[['imd_band_filled']]).flatten()
    
    print(f"   → Filled {missing_count} missing values with 'Unknown' category")
    print(f"   → Encoded as: {dict(zip(imd_order_with_unknown, range(len(imd_order_with_unknown))))} (unknown values: -1)")
    
    encoders['imd_band'] = imd_encoder
    
    # ================================================================
    # 4. AGE_BAND - Ordinal Encoding (Ordinal Variable)
    # ================================================================
    print("\n👥 Processing 'age_band' (Ordinal)")
    
    # Define age band order
    age_order = ['0-35', '35-55', '55<=']  # Adjust based on your actual age bands
    
    unique_ages = X_train_encoded['age_band'].unique()
    print(f"   → Found age bands: {sorted(unique_ages)}")
    
    age_encoder = OrdinalEncoder(categories=[age_order], handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded['age_band_ord'] = age_encoder.fit_transform(X_train_encoded[['age_band']]).flatten()
    
    if X_test_encoded is not None:
        X_test_encoded['age_band_ord'] = age_encoder.transform(X_test_encoded[['age_band']]).flatten()
    
    print(f"   → Encoded as: {dict(zip(age_order, range(len(age_order))))}")
    
    encoders['age_band'] = age_encoder
    
    # ================================================================
    # 5. DISABILITY - Binary Encoding (Binary Variable)
    # ================================================================
    print("\n♿ Processing 'disability' (Binary)")
    
    unique_disability = X_train_encoded['disability'].unique()
    print(f"   → Found disability values: {sorted(unique_disability)}")
    
    # Simple binary encoding (assuming 'Y'/'N' or similar)
    X_train_encoded['disability_binary'] = (X_train_encoded['disability'] == 'Y').astype(int)
    
    if X_test_encoded is not None:
        X_test_encoded['disability_binary'] = (X_test_encoded['disability'] == 'Y').astype(int)
    
    print(f"   → Encoded as: 1 for 'Y', 0 for others")
    
    # ================================================================
    # 6. COMBINE ALL ENCODED FEATURES
    # ================================================================
    print("\n🔗 COMBINING ENCODED FEATURES")
    print("=" * 50)
    
    # Numerical columns to keep
    numerical_cols = ['excellent_Score', 'active_in_VLE', 'student_engagementt', 'courses_per_term']
    
    # Create final feature set
    final_features_train = pd.concat([
        X_train_encoded[numerical_cols],                    # Numerical features
        region_df_train,                                    # One-hot encoded region
        X_train_encoded[['highest_education_ord']],         # Ordinal education
        X_train_encoded[['imd_band_ord']],                  # Ordinal IMD band
        X_train_encoded[['age_band_ord']],                  # Ordinal age band
        X_train_encoded[['disability_binary']]              # Binary disability
    ], axis=1)
    
    if X_test_encoded is not None:
        final_features_test = pd.concat([
            X_test_encoded[numerical_cols],
            region_df_test,
            X_test_encoded[['highest_education_ord']],
            X_test_encoded[['imd_band_ord']],
            X_test_encoded[['age_band_ord']],
            X_test_encoded[['disability_binary']]
        ], axis=1)
    else:
        final_features_test = None

    print(f"{'✅ Final feature count':24}: {final_features_train.shape[1]}")
    print(f"{'   → Numerical features':25}: {len(numerical_cols)}")
    print(f"{'   → Region (one-hot)':25}: {len(region_feature_names)}")
    print(f"{'   → Ordinal features':25}: 3 (education, imd_band, age_band)")
    print(f"{'   → Binary features':25}: 1 (disability)")

    
    return final_features_train, final_features_test, encoders


def print_encoding_summary():
    """Print a summary of encoding strategies used"""
    print("\n📋 ENCODING SUMMARY:")
    print("=" * 30)
    for var, encoder_type in [
        ('region', 'OneHotEncoder'),
        ('highest_education', 'OrdinalEncoder'), 
        ('imd_band', 'OrdinalEncoder'),
        ('age_band', 'OrdinalEncoder'),
        ('disability', 'Binary')
    ]:
        # aquí aplicamos :22 al nombre de la variable
        print(f"   {var:22}: {encoder_type}")

def save_encoded_data(X_train_encoded, X_test_encoded=None, y_train=None, y_test=None):
    """
    Save encoded training and test data along with labels to CSV files.
    
    Parameters:
    - X_train_encoded: Encoded training features (DataFrame)
    - X_test_encoded: Encoded test features (DataFrame, optional)
    - y_train: Training labels (Series, optional)
    - y_test: Test labels (Series, optional)
    """
    # Define the output directory path
    # Since we're in Notebooks folder, go up one level (..) then into Data/output
    output_dir = os.path.join('..', 'Data', 'output')

    # Simple names for features
    train_file_simple = os.path.join(output_dir, 'X_train_encoded.csv')
    test_file_simple = os.path.join(output_dir, 'X_test_encoded.csv')

    # Simple names for labels
    y_train_file = os.path.join(output_dir, 'y_train.csv')
    y_test_file = os.path.join(output_dir, 'y_test.csv')

    try:
        # Save the encoded training data
        X_train_encoded.to_csv(train_file_simple, index=False)
        print(f"✅ Training data saved to: {train_file_simple}")
        print(f"   Shape: {X_train_encoded.shape}")
        
        # Save the training labels (Series type)
        if y_train is not None:
            y_train.to_csv(y_train_file, index=False)
            print(f"✅ Training labels saved to: {y_train_file}")
            print(f"   Shape: {y_train.shape}")
        else:
            print("⚠️  No training labels to save (y_train is None)")
        
        # Save the encoded test data (if it exists)
        if X_test_encoded is not None:
            X_test_encoded.to_csv(test_file_simple, index=False)
            print(f"✅ Test data saved to: {test_file_simple}")
            print(f"   Shape: {X_test_encoded.shape}")
        else:
            print("⚠️  No test data to save (X_test_encoded is None)")
        
        # Save the test labels (Series type)
        if y_test is not None:
            y_test.to_csv(y_test_file, index=False)
            print(f"✅ Test labels saved to: {y_test_file}")
            print(f"   Shape: {y_test.shape}")
        else:
            print("⚠️  No test labels to save (y_test is None)")
        
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        print("Please check that the Data/output directory exists and you have write permissions")
