# =============================================================================
# COURSES PER TERM ANALYSIS
# =============================================================================

import pandas as pd
import sys
import os

# Add Python_files directory to path
sys.path.append('../Python_files')

# Load the processed data
try:
    processed_data = pd.read_csv('../Data/output/processed_data.csv')
    print("✓ Loaded processed data successfully")
except FileNotFoundError:
    print("❌ Processed data file not found. Please run the main analysis first.")
    sys.exit(1)

print("\n" + "="*60)
print("COURSES PER TERM ANALYSIS")
print("="*60)

# Get unique courses per term
courses_per_term = processed_data.groupby('code_presentation')['code_module'].unique()

print("COURSES OFFERED IN EACH TERM:")
print("-" * 40)
for term, courses in courses_per_term.items():
    courses_list = sorted(courses)
    print(f"Term {term}: {', '.join(courses_list)} ({len(courses_list)} courses)")

# Count students per course per term
print(f"\nSTUDENT ENROLLMENT BY COURSE AND TERM:")
print("-" * 50)
course_term_counts = processed_data.groupby(['code_presentation', 'code_module']).size().unstack(fill_value=0)

# Display the enrollment matrix
print(course_term_counts)

# Summary statistics
print(f"\nSUMMARY:")
print(f"  Total terms: {len(courses_per_term)}")
print(f"  Total unique courses: {processed_data['code_module'].nunique()}")
print(f"  Average courses per term: {processed_data.groupby('code_presentation')['code_module'].nunique().mean():.1f}")

# Show which courses are most common across terms
course_frequency = processed_data['code_module'].value_counts()
print(f"\nCOURSE FREQUENCY ACROSS ALL TERMS:")
print("-" * 40)
for course, count in course_frequency.items():
    percentage = count / len(processed_data) * 100
    print(f"  {course}: {count:,} students ({percentage:.1f}%)")

# Show detailed breakdown by term
print(f"\nDETAILED BREAKDOWN BY TERM:")
print("-" * 40)
for term in sorted(processed_data['code_presentation'].unique()):
    term_data = processed_data[processed_data['code_presentation'] == term]
    courses_in_term = term_data['code_module'].value_counts()
    print(f"\nTerm {term}:")
    for course, count in courses_in_term.items():
        percentage = count / len(term_data) * 100
        print(f"  {course}: {count:,} students ({percentage:.1f}%)") 