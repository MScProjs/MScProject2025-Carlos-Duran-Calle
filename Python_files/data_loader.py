"""
Data loading module for Student Assessment Analysis
Handles loading and initial validation of all CSV files.
"""

import pandas as pd
import os
from typing import Dict, Tuple
from config_file import DATA_FOLDER, DATA_FILES


def load_all_data() -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files into pandas DataFrames.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing all loaded DataFrames
    """
    data = {}
    
    print(f"Loading data from: {DATA_FOLDER}")
    
    for key, filename in DATA_FILES.items():
        filepath = os.path.join(DATA_FOLDER, filename)
        try:
            data[key] = pd.read_csv(filepath, sep=',')
            print(f"✓ Loaded {filename}: {data[key].shape}")
        except FileNotFoundError:
            print(f"✗ Error: Could not find {filepath}")
            print(f"   Please ensure the file exists in the Data folder")
            raise
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
            raise
    
    return data


def validate_data(data: Dict[str, pd.DataFrame]) -> bool:
    """
    Perform basic validation on loaded data.
    
    Args:
        data: Dictionary of DataFrames to validate
        
    Returns:
        bool: True if validation passes
    """
    required_columns = {
        'assessments': ['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'date'],
        'student_assessment': ['id_assessment', 'id_student', 'date_submitted', 'is_banked', 'score'],
        'student_registration': ['code_module', 'code_presentation', 'id_student', 'date_registration', 'date_unregistration'],
        'student_vle': ['code_module', 'code_presentation', 'id_student', 'date', 'sum_click'],
        'student_info': ['code_module', 'code_presentation', 'id_student', 'gender', 'region', 'highest_education']
    }
    
    validation_passed = True
    
    for table_name, required_cols in required_columns.items():
        if table_name in data:
            df = data[table_name]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"✗ Missing columns in {table_name}: {missing_cols}")
                validation_passed = False
            else:
                print(f"✓ {table_name} has all required columns")
        else:
            print(f"✗ Table {table_name} not found in loaded data")
            validation_passed = False
    
    return validation_passed


def get_data_info(data: Dict[str, pd.DataFrame]) -> None:
    """
    Print basic information about all loaded datasets.
    
    Args:
        data: Dictionary of DataFrames
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for name, df in data.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"  Missing values: {missing}")
        else:
            print(f"  Missing values: None")


def load_and_validate() -> Tuple[Dict[str, pd.DataFrame], bool]:
    """
    Load all data and perform validation.
    
    Returns:
        Tuple[Dict[str, pd.DataFrame], bool]: Data dictionary and validation status
    """
    print("="*60)
    print("LOADING STUDENT ASSESSMENT DATA")
    print("="*60)
    
    try:
        data = load_all_data()
        
        print("\nValidating data structure...")
        is_valid = validate_data(data)
        
        if is_valid:
            print("\n✓ All data validation checks passed")
            get_data_info(data)
        else:
            print("\n✗ Data validation failed")
            print("Please check your CSV files and ensure they have the required columns")
        
        return data, is_valid
        
    except Exception as e:
        print(f"\n✗ Error during data loading: {e}")
        return {}, False


if __name__ == "__main__":
    # Test the data loading functionality
    print("Testing data loading module...")
    data, valid = load_and_validate()
    
    if valid:
        print("\n✓ Data loading module test successful!")
        print(f"Loaded {len(data)} datasets successfully")
    else:
        print("\n✗ Data loading module test failed!")
        print("Please check your data files and folder structure")