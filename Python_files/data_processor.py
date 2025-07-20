"""
Data processing module for Student Assessment Analysis
Contains functions for data transformation, merging, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set
from config_file import ASSESSMENT_TYPE_FILTER, MERIT_SCORE, STUDENT_INFO_COLUMNS, FINAL_RESULT_MAP


def find_first_assessments(df_assessments: pd.DataFrame) -> pd.DataFrame:
    """
    Find the first TMA assessment for each course.
    
    Args:
        df_assessments: Assessment DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with first assessment information
    """
    print("Finding first assessments...")
    
    # Filter for TMA assessments
    df_tma = df_assessments[df_assessments['assessment_type'] == ASSESSMENT_TYPE_FILTER].copy()
    print(f"  Found {len(df_tma)} TMA assessments")
    
    # For each course, find the earliest TMA date
    earliest_tma = df_tma.groupby(['code_module', 'code_presentation'])['date'].min().reset_index()
    earliest_tma = earliest_tma.rename(columns={'date': 'earliest_tma_date'})
    
    # Merge to get all TMA assessments that are at the earliest date for each course
    df_first_tma = pd.merge(df_tma, earliest_tma, 
                           left_on=['code_module', 'code_presentation', 'date'],
                           right_on=['code_module', 'code_presentation', 'earliest_tma_date'])
    
    # Convert the 'date' column to integer
    df_first_tma['date'] = df_first_tma['date'].astype(int)
    
    # Select only the relevant columns
    df_first_tma = df_first_tma[['code_module', 'code_presentation', 'id_assessment', 'date', 'weight']]

    df_first_tma = pd.DataFrame(df_first_tma)
    
    print(f"✓ Found {len(df_first_tma)} first assessments across {earliest_tma.shape[0]} courses")
    return df_first_tma


def filter_active_students(df_student_registration: pd.DataFrame, 
                         df_first_tma: pd.DataFrame) -> pd.DataFrame:
    """
    Filter students who were active during the first assessment.
    
    Args:
        df_student_registration: Student registration DataFrame
        df_first_tma: First assessment DataFrame
        
    Returns:
        pd.DataFrame: Filtered student registrations
    """
    print("Filtering active students...")
    
    # Merge to get the first assessment date for each student registration
    df_reg_with_tma = pd.merge(
        df_student_registration,
        df_first_tma[['code_module', 'code_presentation', 'id_assessment', 'date']],
        on=['code_module', 'code_presentation'],
        how='left'
    )
    
    print(f"  Students before filtering: {len(df_reg_with_tma):,}")
    
    # Keep students who did not withdraw before the first assessment
    filtered_reg = df_reg_with_tma[
        (df_reg_with_tma['date_unregistration'].isna()) | 
        (df_reg_with_tma['date_unregistration'] >= df_reg_with_tma['date'])
    ].copy()
    
    # Rename the date column to date_first_assessment
    #filtered_reg = filtered_reg.rename(columns={'date': 'date_first_assessment'})
    #filtered_reg.rename(columns={'date': 'date_first_assessment'}, inplace=True)
    filtered_reg.rename(columns={'date': 'date_first_assessment'}, inplace=True)  # type: ignore

    print(f"✓ Active students after filtering: {len(filtered_reg):,}")
    return pd.DataFrame(filtered_reg)


def merge_assessment_scores(filtered_reg: pd.DataFrame, 
                          df_student_assessment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge student registrations with their assessment scores.
    
    Args:
        filtered_reg: Filtered student registrations
        df_student_assessment: Student assessment scores
        
    Returns:
        pd.DataFrame: Merged DataFrame with scores
    """
    print("Merging assessment scores...")
    
    # Merge to get the score and is_banked for the first assessment of each student
    merged_student_assessment = pd.merge(
        filtered_reg,
        df_student_assessment[['id_assessment', 'id_student', 'date_submitted', 'is_banked', 'score']],
        on=['id_assessment', 'id_student'],
        how='left'
    )
    
    # Count students before filling NaN values
    students_with_scores = merged_student_assessment['score'].notna().sum()
    students_without_scores = merged_student_assessment['score'].isna().sum()
    
    print(f"  Students with scores: {students_with_scores:,}")
    print(f"  Students without scores: {students_without_scores:,}")
    
    # Fill NaN values for students who didn't submit
    # score = 0: The student did not earn any points because they did not submit the assessment
    # is_banked = 0: The assessment was not banked (not carried over from a previous presentation)
    merged_student_assessment[['score', 'is_banked']] = merged_student_assessment[['score', 'is_banked']].fillna(0)
    
    print(f"✓ Merged assessment scores for {len(merged_student_assessment):,} students")
    return merged_student_assessment


def calculate_vle_engagement(merged_student_assessment: pd.DataFrame, 
                           df_student_vle: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VLE engagement metrics for each student.
    
    Args:
        merged_student_assessment: DataFrame with student assessments
        df_student_vle: Student VLE interactions
        
    Returns:
        pd.DataFrame: DataFrame with VLE engagement metrics
    """
    print("Calculating VLE engagement...")
    
    # Merge with VLE data
    merged_vle = pd.merge(
        merged_student_assessment.reset_index(),
        df_student_vle,
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )
    
    print(f"  Total VLE interactions before filtering: {len(merged_vle):,}")
    
    # Only keep clicks before or at the first assessment
    merged_vle = merged_vle[
        merged_vle['date'] <= merged_vle['date_first_assessment']
    ]
    
    print(f"  VLE interactions before first assessment: {len(merged_vle):,}")
    
    # Group by the original index and sum sum_click
    total_clicks = merged_vle.groupby('index')['sum_click'].sum()
    
    # Assign to merged_student_assessment
    merged_student_assessment['total_click_vle'] = merged_student_assessment.index.map(total_clicks).fillna(0).astype(int)
    
    # Calculate the average total_click for each course and presentation
    average_clicks = (
        merged_student_assessment
        .groupby(['code_module', 'code_presentation'])['total_click_vle']
        .mean()
        .rename('average_click_vle') # type: ignore
    )
    
    # Map the average to each row
    merged_student_assessment['average_click_vle'] = merged_student_assessment.set_index(['code_module', 'code_presentation']).index.map(average_clicks)
    merged_student_assessment['average_click_vle'] = merged_student_assessment['average_click_vle'].round(1)
    
    # Count students with no VLE activity
    no_vle_students = (merged_student_assessment['total_click_vle'] == 0).sum()
    print(f"  Students with no VLE activity: {no_vle_students:,}")
    
    print(f"✓ Calculated VLE engagement for {len(merged_student_assessment):,} students")
    return merged_student_assessment


def create_engagement_features(merged_student_assessment: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement features based on score and VLE activity.
    
    Args:
        merged_student_assessment: DataFrame with assessment and VLE data
        
    Returns:
        pd.DataFrame: DataFrame with engagement features
    """
    print("Creating engagement features...")
    
    # Add 'excellent_Score' column: 1 if score >= merit_score, else 0
    merged_student_assessment['excellent_Score'] = (
        merged_student_assessment['score'] >= MERIT_SCORE
    ).astype(int)
    
    excellent_students = merged_student_assessment['excellent_Score'].sum()
    print(f"  Students with excellent scores (>= {MERIT_SCORE}): {excellent_students:,}")
    
    # Add 'active_in_VLE' column: 1 if total_click_vle > average_click_vle, else 0
    merged_student_assessment['active_in_VLE'] = (
        merged_student_assessment['total_click_vle'] > merged_student_assessment['average_click_vle']
    ).astype(int)
    
    active_students = merged_student_assessment['active_in_VLE'].sum()
    print(f"  Students active in VLE: {active_students:,}")
    
    # Add 'student_engagementt' column: 1 if either excellent_Score or active_in_VLE is 1, else 0
    merged_student_assessment['student_engagementt'] = (
        (merged_student_assessment['excellent_Score'] | merged_student_assessment['active_in_VLE'])
    ).astype(int)
    
    engaged_students = merged_student_assessment['student_engagementt'].sum()
    print(f"  Overall engaged students: {engaged_students:,}")
    
    print("✓ Created engagement features")
    return merged_student_assessment


def merge_student_info(student_first_assessment: pd.DataFrame, 
                      df_student_info: pd.DataFrame) -> pd.DataFrame:
    """
    Merge student assessment data with student demographic information.
    
    Args:
        student_first_assessment: DataFrame with first assessment data
        df_student_info: Student demographic information
        
    Returns:
        pd.DataFrame: Complete merged DataFrame
    """
    print("Merging student demographic information...")
    
    # Merge on code_module, code_presentation, and id_student
    student_first_assessment_merged = pd.merge(
        student_first_assessment,
        df_student_info[STUDENT_INFO_COLUMNS],
        on=['code_module', 'code_presentation', 'id_student'],
        how='left'
    )
    
    # Map final_result to final_result_code
    student_first_assessment_merged['final_result_code'] = student_first_assessment_merged['final_result'].map(FINAL_RESULT_MAP) # type: ignore
    
    # Check for missing demographic data
    missing_demo = student_first_assessment_merged[STUDENT_INFO_COLUMNS[3:]].isnull().sum().sum()
    if missing_demo > 0:
        print(f"  Warning: {missing_demo} missing demographic values found")
    
    print(f"✓ Final merged dataset: {student_first_assessment_merged.shape}")
    
    # Show final result distribution
    result_dist = student_first_assessment_merged['final_result'].value_counts()
    print("  Final result distribution:")
    for result, count in result_dist.items():
        percentage = count / len(student_first_assessment_merged) * 100
        print(f"    {result}: {count:,} ({percentage:.1f}%)")
    
    return student_first_assessment_merged


def get_students_with_missing_registration(filtered_reg: pd.DataFrame) -> Set[int]:
    """
    Get set of students with missing registration dates.
    
    Args:
        filtered_reg: Filtered registration DataFrame
        
    Returns:
        Set[int]: Set of student IDs with missing registration dates
    """
    missing_students = set(filtered_reg[filtered_reg['date_registration'].isna()]['id_student'])
    if missing_students:
        print(f"⚠ Found {len(missing_students)} students with missing registration dates")
    else:
        print("✓ All students have registration dates")
    return missing_students


def process_all_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Process all data through the complete pipeline.
    
    Args:
        data: Dictionary containing all raw DataFrames
        
    Returns:
        pd.DataFrame: Fully processed DataFrame ready for analysis
    """
    print("\n" + "="*60)
    print("STARTING DATA PROCESSING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Find first assessments
        df_first_tma = find_first_assessments(data['assessments'])
        
        # Step 2: Filter active students
        filtered_reg = filter_active_students(data['student_registration'], df_first_tma)
        
        # Step 3: Check for missing registration dates
        missing_students = get_students_with_missing_registration(filtered_reg)
        
        # Step 4: Merge assessment scores
        merged_student_assessment = merge_assessment_scores(filtered_reg, data['student_assessment'])
        
        # Step 5: Calculate VLE engagement
        merged_student_assessment = calculate_vle_engagement(merged_student_assessment, data['student_vle'])
        
        # Step 6: Create engagement features
        merged_student_assessment = create_engagement_features(merged_student_assessment)
        
        # Step 7: Merge with student info
        final_data = merge_student_info(merged_student_assessment, data['student_info'])
        
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Final dataset shape: {final_data.shape}")
        print(f"Columns: {len(final_data.columns)}")
        print(f"Memory usage: {final_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return final_data
        
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}")
        raise


if __name__ == "__main__":
    print("Data processing module ready for import")
    print("Use process_all_data(data_dict) to run the complete pipeline")