"""
Configuration file for Student Assessment Analysis
Contains all constants and configuration parameters used across the analysis.
"""

import os

# Data configuration
DATA_FOLDER = os.path.join('..', 'Data')  # Relative path from Python_files folder
#DATA_FOLDER = 'Data'

# File names
DATA_FILES = {
    'assessments': 'assessments.csv',
    'courses': 'courses.csv',
    'student_assessment': 'studentAssessment.csv',
    'student_info': 'studentInfo.csv',
    'student_registration': 'studentRegistration.csv',
    'student_vle': 'studentVle.csv',
    'vle': 'vle.csv'
}

# Analysis parameters
MERIT_SCORE = 70
ASSESSMENT_TYPE_FILTER = 'TMA'

# Categorical variables for cross-table analysis
CATEGORICAL_VARS = [
    'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability'
]

# Student info columns to include in analysis
STUDENT_INFO_COLUMNS = [
    'code_module', 'code_presentation', 'id_student',
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'studied_credits', 'disability', 'final_result'
]

# Final result mapping
FINAL_RESULT_MAP = {
    'Withdrawn': 0, 
    'Fail': 1, 
    'Pass': 2, 
    'Distinction': 2
}

# Result code to name mapping
RESULT_CODE_MAPPING = {
    0: 'Withdrawn',
    1: 'Fail', 
    2: 'Pass'
}

# Visualization configuration
PLOT_COLORS = ['#d62728', '#ff7f0e', '#1f77b4']  # Red (Withdrawn), Orange (Fail), Blue (Pass)
PLOT_STYLE = 'seaborn-v0_8'

# Variable configurations for plotting
VARIABLE_CONFIGS = {
    'gender': {
        'x_values': [0, 1],
        'x_label': 'Gender (Female=0, Male=1)',
        'categories': ['Female', 'Male']
    },
    'age_band': {
        'x_values': [0, 1, 2],
        'x_label': 'Age Band (0: 0-35, 1: 35-55, 2: 55+)',
        'categories': ['0-35', '35-55', '55+']
    },
    'disability': {
        'x_values': [0, 1],
        'x_label': 'Disability Status (No=0, Yes=1)',
        'categories': ['No', 'Yes']
    },
    'highest_education': {
        'x_values': [0, 1, 2, 3, 4],
        'x_label': 'Education Level (0: No Formal → 4: Post Graduate)',
        'categories': [
            'No Formal Qualifications',
            'Lower Than A Level', 
            'A Level or Equivalent',
            'Higher Education Qualification',
            'Post Graduate Qualification'
        ],
        'order': [
            'No Formal quals',
            'Lower Than A Level', 
            'A Level or Equivalent',
            'HE Qualification',
            'Post Graduate Qualification'
        ]
    },
    'imd_band': {
        'x_values': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        'x_label': 'Income Deprivation Band (% midpoint)',
        'categories': ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                      '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    },
    'region': {
        'x_values': None,  # Will be set dynamically
        'x_label': 'Region (Alphabetical Order)',
        'categories': None  # Will be set dynamically
    }
}