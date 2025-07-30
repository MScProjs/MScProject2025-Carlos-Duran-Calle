# MScProject - Student Assessment Analysis

A comprehensive, modular student assessment analysis system refactored from Jupyter notebook into maintainable Python modules with interactive visualizations and statistical analysis.

## 🎯 Project Overview

This project analyzes student performance data to identify factors affecting academic outcomes. It examines relationships between demographic characteristics, VLE engagement, assessment scores, and final results through cross-tabulation analysis and linear regression.

### Key Research Questions
- How do demographic factors influence student success?
- What is the relationship between VLE engagement and academic performance?
- Which variables are the strongest predictors of student outcomes?
- How do different student segments perform across courses?

## 📁 Project Structure

```
MScProject/
├── README.md                     # This comprehensive guide
├── Data/                         # 📊 Data files and outputs
│   ├── assessments.csv           # Assessment information
│   ├── courses.csv               # Course details
│   ├── studentAssessment.csv     # Student assessment scores
│   ├── studentInfo.csv           # Student demographics
│   ├── studentRegistration.csv   # Registration records
│   ├── studentVle.csv            # VLE interaction data
│   ├── vle.csv                   # VLE resource information
│   ├── output/                   # 📁 Generated output files
│   │   ├── processed_data.csv    # Processed dataset
│   │   ├── X_test_encoded.csv    # X Test data encoded
│   │   ├── X_train_encoded.csv   # X Train data encoded
│   │   ├── y_test.csv   		  # Y Test data encoded
│   │   └── y_train.csv	  		  # Y Train data encoded
│   │
│   └── model_metrics/            # 🤖 Model outputs and evaluation
│       ├── models/               # 📁 Trained model files
│       │   ├── knn_optimized.pkl
│       │   ├── knn_optimized_scaler.pkl
│       │   ├── random_forest_optimized.pkl
│       │   ├── multinomial_logistic_regression_optimized.pkl
│       │   ├── lightgbm_optimized.pkl                    # 🌳 LightGBM model
│       │   ├── svm_optimized.pkl                         # 🎯 SVM model
│       │   └── svm_optimized_scaler.pkl                  # 🔧 SVM scaler (required)
│       ├── metrics/              # 📁 Performance metrics and data
│       │   ├── knn_optimized_metrics.json
│       │   ├── knn_optimized_confusion_matrix.json
│       │   ├── random_forest_optimized_metrics.json
│       │   ├── random_forest_optimized_confusion_matrix.json
│       │   ├── random_forest_optimized_confusion_matrix.txt
│       │   ├── random_forest_optimized_feature_importance.csv
│       │   ├── random_forest_optimized_feature_importance.json
│       │   ├── multinomial_logistic_regression_optimized_metrics.json
│       │   ├── multinomial_logistic_regression_optimized_confusion_matrix.json
│       │   ├── multinomial_logistic_regression_optimized_confusion_matrix.txt
│       │   ├── multinomial_logistic_regression_optimized_coefficients.csv
│       │   ├── multinomial_logistic_regression_optimized_coefficients.json
│       │   ├── lightgbm_optimized_metrics.json           # 🌳 LightGBM metrics
│       │   ├── lightgbm_optimized_confusion_matrix.json  # 🌳 LightGBM confusion matrix
│       │   ├── lightgbm_optimized_feature_importance.csv # 🌳 LightGBM feature importance
│       │   ├── svm_optimized_metrics.json                # 🎯 SVM metrics
│       │   └── svm_optimized_confusion_matrix.json       # 🎯 SVM confusion matrix
│       ├── reports/              # 📁 Classification reports
│       │   ├── knn_optimized_classification_report.json
│       │   ├── random_forest_optimized_classification_report.json
│       │   ├── random_forest_optimized_classification_report.txt
│       │   ├── multinomial_logistic_regression_optimized_classification_report.json
│       │   ├── multinomial_logistic_regression_optimized_classification_report.txt
│       │   ├── lightgbm_optimized_classification_report.json      # 🌳 LightGBM report
│       │   └── svm_optimized_classification_report.json           # 🎯 SVM report
│       ├── knn_optimized_USAGE_INSTRUCTIONS.txt
│       ├── random_forest_optimized_USAGE_INSTRUCTIONS.txt
│       ├── multinomial_logistic_regression_optimized_USAGE_INSTRUCTIONS.txt
│       ├── lightgbm_optimized_USAGE_INSTRUCTIONS.txt              # 🌳 LightGBM instructions
│       └── svm_optimized_USAGE_INSTRUCTIONS.txt                   # 🎯 SVM instructions
│
├── Python_files/                 # 🐍 Core analysis modules
│   ├── config.py                 # Configuration and constants
│   ├── data_loader.py            # Data loading and validation
│   ├── data_processor.py         # Data processing pipeline
│   ├── analysis_engine.py        # Statistical analysis functions
│   ├── encoding_utils.py         # Encoding for the training and test data
│   └── visualization.py          # Interactive plotting
└── Notebooks/                    # 📓 Analysis interfaces
    ├── 01_data_ingest_cleaning.ipynb              # Complete analysis workflow
    ├── 02_visual_analysis.ipynb                   # Visualisation of the data
    ├── 03_data_stratification_encoding.ipynb      # Data stratification and encoding
    ├── 04_model_random_forest.ipynb               # Random Forest model optimization
    ├── 05_model_multi_logistic_regression.ipynb   # Logistic Regression model optimization
    ├── 06_model_knn.ipynb                         # K-Nearest Neighbors model optimization
    ├── 07_model_lightGBM.ipynb                    # 🌳 LightGBM model optimization
    ├── 08_model_SVM.ipynb                         # SVM model optimization
    └── X0_model_comparison_analysis.ipynb         # Model Comparison

```

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter

# Optional for enhanced features
pip install openpyxl xlsxwriter kaleido
```

### 2. Data Setup
Place your CSV files in the `Data/` folder:
- ✅ `assessments.csv` - Assessment metadata
- ✅ `courses.csv` - Course information  
- ✅ `studentAssessment.csv` - Student scores
- ✅ `studentInfo.csv` - Demographics
- ✅ `studentRegistration.csv` - Enrollment data
- ✅ `studentVle.csv` - VLE interactions
- ✅ `vle.csv` - VLE resource data

### 3. Choose Your Analysis Approach

#### 🎯 **Option A: Complete Analysis (Recommended for first-time users)**
```bash
# Open Jupyter Notebook
jupyter notebook

# Navigate to and open: Notebooks/main_analysis.ipynb
# Run all cells for complete analysis pipeline
```

#### ⚡ **Option B: Quick Analysis (For exploration and specific tasks)**
```bash
# Open: Notebooks/quick_analysis.ipynb
# Run setup cell, then choose specific analysis sections
```

#### 🔧 **Option C: Custom Analysis (For advanced users)**
```python
# Import modules in your own code
import sys
sys.path.append('Python_files')

from data_loader import load_and_validate
from data_processor import process_all_data
from analysis_engine import run_complete_analysis
from visualization import create_all_visualizations

# Run custom analysis
data, valid = load_and_validate()
processed_data = process_all_data(data)
cross_tables, summary_stats, insights, segments = run_complete_analysis(processed_data)
figures = create_all_visualizations(cross_tables, summary_stats, segments)
```

## 🔧 Module Documentation

### 📋 `Python_files/config.py` - Configuration Hub
**Purpose**: Centralized configuration management
- **Data paths**: Automatic path resolution between folders
- **Analysis parameters**: Merit score (70), assessment type (TMA)
- **Variables**: Categorical variables for cross-tabulation
- **Visualization**: Colors, styles, and plotting configurations
- **Mappings**: Variable configurations for x-axis values and labels

### 📥 `Python_files/data_loader.py` - Data Management
**Purpose**: Robust data loading with validation
- `load_all_data()` - Load all CSV files with error handling
- `validate_data()` - Check required columns and data integrity
- `get_data_info()` - Display comprehensive dataset summaries
- `load_and_validate()` - Complete pipeline with validation feedback

**Key Features**:
- ✅ Automatic file detection and loading
- 🔍 Data quality validation
- 📊 Memory usage reporting
- ⚠️ Clear error messages and troubleshooting

### 🔄 `Python_files/data_processor.py` - Data Pipeline
**Purpose**: Transform raw data into analysis-ready format
- `find_first_assessments()` - Identify first TMA assessment per course
- `filter_active_students()` - Remove students who withdrew early
- `merge_assessment_scores()` - Combine registration and assessment data
- `calculate_vle_engagement()` - Compute VLE interaction metrics
- `create_engagement_features()` - Generate binary engagement indicators
- `merge_student_info()` - Add demographic information
- `process_all_data()` - Complete processing pipeline

**Data Processing Steps**:
1. 🎯 **Assessment Identification**: Find first TMA for each course
2. 👥 **Student Filtering**: Keep only active students during assessment
3. 📊 **Score Integration**: Merge assessment scores (0 for non-submissions)
4. 🖱️ **VLE Analysis**: Calculate total and relative VLE engagement
5. 🏆 **Feature Engineering**: Create performance and engagement flags
6. 📋 **Demographics**: Add student background information

### 📊 `Python_files/analysis_engine.py` - Statistical Analysis
**Purpose**: Comprehensive statistical analysis and insights
- `create_cross_tables()` - Generate proportion tables for categorical variables
- `calculate_linear_regression()` - Perform regression analysis with error handling
- `analyze_cross_table_trends()` - Analyze trends across all variables
- `analyze_student_segments()` - Identify and analyze student groups
- `run_complete_analysis()` - Execute full analysis pipeline

**Statistical Methods**:
- 📈 **Cross-tabulation**: Row-normalized proportions by outcome
- 📉 **Linear Regression**: Trend analysis with R² and slope calculations
- 👥 **Student Segmentation**: High performers, engaged students, at-risk groups
- 🎯 **Correlation Analysis**: Relationship strength assessment

### 📈 `Python_files/visualization.py` - Interactive Visualizations
**Purpose**: Create publication-ready interactive plots
- `create_plotly_regression_plot()` - Individual variable regression plots
- `plot_all_cross_tables()` - Generate all regression visualizations
- `create_student_segments_plot()` - Student segment comparison charts
- `create_correlation_heatmap()` - R² correlation matrix
- `save_all_plots()` - Export plots to HTML files
- `create_all_visualizations()` - Complete visualization pipeline

**Visualization Features**:
- 🎨 **Interactive Plots**: Zoom, pan, hover for detailed exploration
- 📊 **Regression Analysis**: Scatter plots with trend lines and statistics
- 👥 **Segment Comparison**: Bar charts for student group analysis
- 🔥 **Heatmaps**: Correlation strength visualization
- 💾 **Export Options**: HTML files for sharing and presentations

## 📓 Notebook Interfaces

### 🎯 `Notebooks/main_analysis.ipynb` - Complete Workflow
**Best for**: First-time users, comprehensive reports, complete analysis

**Features**:
- 📋 **Step-by-step pipeline**: Guided analysis from data loading to results
- 🔍 **Data quality checks**: Comprehensive validation and summaries
- 📊 **Complete analysis**: All statistical tests and visualizations
- 💾 **Automatic export**: Results saved to files automatically
- 📄 **Executive summary**: Final report with key findings

**Workflow**:
1. ✅ Setup and imports
2. 📥 Data loading and validation
3. 🔄 Data processing pipeline
4. 📊 Statistical analysis
5. 📈 Interactive visualizations
6. 💾 Results export
7. 📋 Summary report

### ⚡ `Notebooks/quick_analysis.ipynb` - Modular Interface
**Best for**: Exploration, specific questions, iterative analysis

**Features**:
- 🎯 **Modular design**: Run only what you need
- ⚡ **Fast iteration**: Quick exploration and testing
- 🔍 **Focused analysis**: Individual variables and plots
- 📊 **Interactive exploration**: Customizable parameters
- 💾 **Quick export**: Essential results only

**Key Sections**:
- 🔧 **Setup**: Always run first
- 📥 **Quick Load**: One-step data processing
- 📋 **Show Data**: Dataset overview and statistics
- 📊 **Cross Tables**: Individual or all cross-tabulations
- 📈 **Individual Plots**: Single variable visualizations
- 📊 **All Plots**: Complete regression analysis
- 🔍 **Custom Exploration**: Flexible data investigation
- 💾 **Quick Export**: Save key results

### 🆕 **New Features Added**:
- 📈 **Show All Plots Cell**: Display all linear regression plots at once
- 💾 **Save Processed Data**: Export processed dataset to `Data/output/`
- 🎯 **Enhanced Statistics**: Detailed regression analysis with trend classification
- 📊 **Interactive Summaries**: Real-time analysis feedback

## 📊 Analysis Features & Capabilities

### 🎯 Student Outcome Analysis
- **Final Results**: Withdrawn (0), Fail (1), Pass/Distinction (2)
- **Performance Metrics**: Score distributions, pass rates, completion rates
- **Engagement Indicators**: VLE activity, assessment submission patterns

### 👥 Demographic Analysis
- **Gender**: Male vs. Female outcomes
- **Age Groups**: 0-35, 35-55, 55+ performance comparison
- **Education Level**: No formal quals → Postgraduate progression
- **Disability Status**: Accessibility and support impact
- **Socioeconomic Background**: Income deprivation band analysis
- **Geographic Factors**: Regional performance variations

### 📈 Statistical Methods
- **Cross-tabulation**: Proportion analysis with row normalization
- **Linear Regression**: Trend identification with R² strength assessment
- **Correlation Analysis**: Variable relationship mapping
- **Segment Analysis**: Student group identification and comparison

### 🎯 Student Segmentation
- **High Performers**: Score ≥ 70 (merit threshold)
- **VLE Engaged**: Above-average platform interaction
- **Overall Engaged**: High performance OR high VLE activity
- **At-Risk Groups**: No VLE activity, low performers
- **Success Predictors**: Combined performance indicators

## 📈 Key Outputs & Results

### 📊 Automatically Generated Files

#### `analysis_results/` Directory
- 📋 **`regression_summary.csv`** - Complete R² and slope statistics
- 📊 **`cross_tables.xlsx`** - All cross-tabulation tables by variable
- 👥 **`student_segments.csv`** - Student group analysis results
- 💡 **`key_insights.txt`** - Narrative summary of key findings
- 📄 **`processed_data_sample.csv`** - Sample of processed dataset

#### `plots/` Directory
- 📈 **Variable regression plots** - Interactive HTML visualizations
- 👥 **`student_segments.html`** - Segment comparison charts
- 🔥 **`correlation_heatmap.html`** - R² correlation matrix
- 🎯 **Individual variable plots** - Detailed trend analysis

#### `Data/output/` Directory
- 💾 **`processed_data.csv`** - Complete processed dataset
- 🕐 **Timestamped versions** - Backup copies with date/time

### 📊 Statistical Outputs
- **R² Values**: Linear relationship strength (0-1 scale)
- **Slope Coefficients**: Trend direction and magnitude
- **Trend Classification**: Strong (>0.7), Moderate (0.4-0.7), Weak (<0.4)
- **Student Segments**: Size, characteristics, and success rates
- **Correlation Rankings**: Strongest predictive variables

### 💡 Insights Generated
- **Best Predictors**: Variables with highest R² values
- **Consistent Trends**: Variables showing uniform patterns
- **Risk Factors**: Characteristics associated with withdrawal
- **Success Indicators**: Factors promoting completion
- **Demographic Patterns**: Group-specific performance trends

## ⚙️ Customization & Configuration

### 🔧 Modifying Analysis Parameters
Edit `Python_files/config.py`:
```python
# Analysis thresholds
MERIT_SCORE = 70                    # Change performance threshold
ASSESSMENT_TYPE_FILTER = 'TMA'      # Modify assessment type

# Add new categorical variables
CATEGORICAL_VARS.append('new_variable')

# Customize plot colors
PLOT_COLORS = ['#custom', '#colors', '#here']
```

### 📊 Adding New Variables
1. **Update config**: Add to `CATEGORICAL_VARS` list
2. **Configure plotting**: Define `VARIABLE_CONFIGS` entry
3. **Update columns**: Modify `STUDENT_INFO_COLUMNS` if needed

Example:
```python
# In config.py
CATEGORICAL_VARS.append('employment_status')

VARIABLE_CONFIGS['employment_status'] = {
    'x_values': [0, 1, 2],
    'x_label': 'Employment (0: Unemployed, 1: Part-time, 2: Full-time)',
    'categories': ['Unemployed', 'Part-time', 'Full-time']
}
```

### 🎨 Visualization Customization
```python
# Custom regression plot
from visualization import create_plotly_regression_plot

custom_plot = create_plotly_regression_plot(
    cross_table=your_table,
    variable_name='Custom Variable',
    x_values=[0, 1, 2],
    x_label='Custom X-axis Label',
    categories=['Cat1', 'Cat2', 'Cat3']
)
```

### 📊 Custom Analysis Examples
```python
# Custom student segment
high_achievers = processed_data[
    (processed_data['score'] >= 80) & 
    (processed_data['total_click_vle'] > processed_data['average_click_vle'])
]

# Custom cross-tabulation
custom_crosstab = pd.crosstab(
    processed_data['custom_variable'],
    processed_data['final_result_code'],
    normalize='index'
)

# Custom correlation analysis
correlation_matrix = processed_data[numeric_columns].corr()
```

## 🛠️ Troubleshooting Guide

### 🔍 Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **FileNotFoundError** | `Data file not found` | ✅ Check all CSV files are in `Data/` folder |
| **Import Errors** | `Module not found` | ✅ Run setup cell first, install required packages |
| **Missing Columns** | `Column 'x' not found` | ✅ Verify CSV file structure matches requirements |
| **Path Issues** | `Cannot find Python_files` | ✅ Run notebooks from `Notebooks/` folder |
| **Memory Issues** | `Out of memory` | ✅ Use data sampling, close other applications |
| **Plot Not Showing** | `Empty visualization` | ✅ Check cross_tables exist, run processing first |

### ⚠️ Data Validation Checklist
- [ ] All 7 CSV files present in `Data/` folder
- [ ] Files contain required columns (see validation output)
- [ ] Data types are appropriate (dates, numbers, strings)
- [ ] No completely empty files
- [ ] File permissions allow reading

### 🚨 Error Messages & Fixes

**"processed_data not found"**
```python
# Solution: Run data processing first
data, valid = load_and_validate()
processed_data = process_all_data(data)
```

**"cross_tables not found"**
```python
# Solution: Run analysis first
cross_tables = create_cross_tables(processed_data)
```

**"Permission denied when saving"**
```python
# Solution: Check folder permissions or change output directory
output_dir = 'alternative_output_folder'
```

### 📈 Performance Optimization
- **Large datasets**: Use `processed_data.sample(n=10000)` for testing
- **Memory management**: Close unused variables with `del variable_name`
- **Plot performance**: Reduce data points or create static plots
- **File size**: Use compression for large exports

## 📊 Advanced Usage & Extensions

### 🔬 Advanced Statistical Analysis
```python
# Time series analysis
temporal_trends = processed_data.groupby('code_presentation').agg({
    'score': 'mean',
    'total_click_vle': 'mean',
    'final_result_code': lambda x: (x >= 2).mean()
})

# Course-specific analysis
course_performance = processed_data.groupby('code_module').agg({
    'excellent_Score': 'mean',
    'active_in_VLE': 'mean',
    'student_engagementt': 'mean'
})

# Interaction effects
interaction_analysis = processed_data.groupby(['gender', 'age_band']).agg({
    'score': ['mean', 'std'],
    'final_result_code': lambda x: (x >= 2).mean()
})
```

### 🤖 Machine Learning Integration
```python
# Prepare data for ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Feature selection
features = ['score', 'total_click_vle', 'excellent_Score', 'active_in_VLE']
X = processed_data[features]
y = processed_data['final_result_code']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 📊 Custom Visualizations
```python
# Custom heatmap
import seaborn as sns

plt.figure(figsize=(10, 8))
correlation_matrix = processed_data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Variable Correlations')
plt.show()

# Custom time series plot
import plotly.express as px

fig = px.line(temporal_trends.reset_index(), 
              x='code_presentation', 
              y='score',
              title='Average Score Over Time')
fig.show()
```

### 📈 Reporting & Export
```python
# Generate executive summary
summary_report = f"""
STUDENT ASSESSMENT ANALYSIS SUMMARY
=====================================

Dataset: {len(processed_data):,} students
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

Key Findings:
- High Performers: {(processed_data['excellent_Score']==1).mean()*100:.1f}%
- VLE Engaged: {(processed_data['active_in_VLE']==1).mean()*100:.1f}%
- Overall Success Rate: {(processed_data['final_result_code']>=2).mean()*100:.1f}%

Strongest Predictors:
{summary_stats.nlargest(3, 'R_Squared')[['Variable', 'Outcome', 'R_Squared']].to_string()}
"""

# Save report
with open('executive_summary.txt', 'w') as f:
    f.write(summary_report)
```

## 🎯 Best Practices & Recommendations

### 📋 Analysis Workflow
1. **Start Small**: Use `quick_analysis.ipynb` for initial exploration
2. **Validate Data**: Always check data quality before analysis
3. **Iterative Approach**: Explore variables individually before comprehensive analysis
4. **Document Findings**: Save key insights and visualizations
5. **Version Control**: Keep timestamped copies of processed data

### 📊 Interpretation Guidelines
- **R² > 0.7**: Strong linear relationship, high predictive value
- **R² 0.4-0.7**: Moderate relationship, useful for insights
- **R² < 0.4**: Weak relationship, limited predictive power
- **Positive slopes**: Higher values → better outcomes
- **Negative slopes**: Higher values → worse outcomes

### 🔍 Data Quality Considerations
- **Missing data**: Understand why data is missing
- **Outliers**: Investigate extreme values
- **Sample sizes**: Ensure adequate representation
- **Temporal effects**: Consider time-based patterns
- **Selection bias**: Account for student filtering effects

### 📈 Visualization Best Practices
- **Interactive plots**: Use for exploration and presentations
- **Static plots**: Use for reports and publications
- **Color consistency**: Maintain color schemes across plots
- **Clear labels**: Include units and explanations
- **Accessibility**: Ensure plots are readable for all audiences

## 🚀 Future Enhancements & Roadmap

### 📊 Planned Features
- [ ] **Advanced Statistics**: Chi-square tests, ANOVA, effect sizes
- [ ] **Machine Learning**: Predictive modeling and feature importance
- [ ] **Time Series**: Temporal pattern analysis
- [ ] **Clustering**: Student group discovery algorithms
- [ ] **Automated Reporting**: PDF report generation
- [ ] **Real-time Dashboard**: Interactive web interface

### 🔧 Technical Improvements
- [ ] **Performance**: Optimization for large datasets
- [ ] **Testing**: Unit tests for all modules
- [ ] **Documentation**: Extended API documentation
- [ ] **Configuration**: GUI-based parameter setting
- [ ] **Integration**: Database connectivity options

### 📈 Analysis Extensions
- [ ] **Predictive Analytics**: Early warning systems
- [ ] **Intervention Analysis**: Support strategy effectiveness
- [ ] **Comparative Studies**: Cross-institutional analysis
- [ ] **Longitudinal Analysis**: Student journey tracking
- [ ] **Resource Optimization**: Learning material effectiveness

## 📞 Support & Community

### 🆘 Getting Help
1. **Check Documentation**: Review this README and module docstrings
2. **Validate Data**: Run data quality checks first
3. **Check Examples**: Review notebook cells for usage patterns
4. **Error Messages**: Read error messages carefully for guidance

### 🐛 Bug Reports & Feature Requests
When reporting issues, include:
- Error messages (full traceback)
- Data characteristics (size, columns, types)
- System information (Python version, OS)
- Steps to reproduce the problem

### 💡 Contributing
- **Code improvements**: Optimize functions, add features
- **Documentation**: Enhance explanations and examples
- **Testing**: Add test cases and validation
- **Visualizations**: Create new plot types
- **Analysis methods**: Implement additional statistical tests

## 📚 References & Acknowledgments

### 📖 Methodology References
- Cross-tabulation analysis for categorical data
- Linear regression for trend analysis
- Student engagement metrics in VLE systems
- Demographic factors in educational outcomes

### 🛠️ Technical Stack
- **Python 3.11+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Statistical analysis
- **Matplotlib/Seaborn**: Additional plotting
- **Jupyter**: Interactive development environment

### 🎯 Dataset Context
This analysis framework is designed for educational data analysis, specifically student assessment and engagement data from learning management systems. The methods are generalizable to similar educational datasets.

---

**📧 Questions or Issues?** 
Check the troubleshooting section, validate your data setup, and review error messages for specific guidance. This modular framework provides a robust foundation for educational data analysis with extensive customization options.

**🎉 Happy Analyzing!**