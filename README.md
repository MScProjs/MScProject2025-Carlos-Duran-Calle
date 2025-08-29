# MScProject - Student Assessment Analysis

A comprehensive machine learning analysis system for student performance prediction using demographic data, VLE engagement metrics, and assessment scores.

## 🎯 Project Overview

This project analyzes student performance data to predict academic outcomes using machine learning models. It examines relationships between demographic characteristics, Virtual Learning Environment (VLE) engagement, assessment scores, and final results through multiple classification models.

### Key Research Questions
- How do demographic factors influence student success?
- What is the relationship between VLE engagement and academic performance?
- Which variables are the strongest predictors of student outcomes?
- How do different machine learning models compare in predicting student results?

## 📁 Project Structure

```
MScProject/
├── README.md                           # Project documentation
├── Data/                               # 📊 Dataset and processed outputs
│   ├── assessments.csv                 # Assessment metadata
│   ├── courses.csv                     # Course information
│   ├── studentAssessment.csv           # Student assessment scores
│   ├── studentInfo.csv                 # Student demographics
│   ├── studentRegistration.csv         # Course registration data
│   ├── studentVle.csv                  # VLE interaction logs
│   ├── vle.csv                         # VLE resource information
│   ├── output/                         # 📁 Processed training/test data
│   │   ├── processed_data.csv          # Cleaned dataset
│   │   ├── X_train_encoded.csv         # Training features
│   │   ├── X_test_encoded.csv          # Test features
│   │   ├── y_train.csv                 # Training labels
│   │   └── y_test.csv                  # Test labels
│   └── model_metrics/                  # 🤖 Model outputs and evaluation
│       ├── models/                     # 📁 Trained model files (.pkl)
│       │   ├── random_forest_optimized.pkl
│       │   ├── multinomial_logistic_regression_optimized.pkl
│       │   ├── knn_optimized.pkl
│       │   ├── lightgbm_optimized.pkl
│       │   ├── svm_optimized.pkl
│       │   ├── neural_network_optimized.pkl
│       │   └── *_scaler.pkl            # Feature scalers for applicable models
│       ├── metrics/                    # 📁 Performance metrics (JSON/CSV)
│       │   ├── *_metrics.json          # Model performance scores
│       │   ├── *_confusion_matrix.json # Confusion matrices
│       │   ├── *_coefficients.json     # Model coefficients/feature importance
│       │   └── *_feature_importance.csv
│       ├── reports/                    # 📁 Classification reports (JSON)
│       └── *_USAGE_INSTRUCTIONS.txt    # Model usage guides
│
├── Documentation/                      # 📋 Project documentation
│   └── Project_Plan-Carlos_Duran.pdf   # Project planning document
│
├── Dissertation_report/                # 📄 Dissertation and research documentation
│   └── [Dissertation files]            # Research papers, reports, and analysis documents
│
├── Python_files/                      # 🐍 Core analysis modules
│   ├── __init__.py                     # Package initialization
│   ├── config_file.py                  # Configuration settings
│   ├── data_loader.py                  # Data loading utilities
│   ├── data_processor.py               # Data cleaning and preprocessing
│   ├── encoding_utils.py               # Feature encoding for ML models
│   ├── analysis_engine.py              # Statistical analysis functions
│   └── visualization.py                # Plotting and visualization tools
│
├── Notebooks/                          # 📓 Analysis workflow notebooks
│   ├── 01_data_ingest_cleaning.ipynb           # Data ingestion and cleaning
│   ├── 02_visual_analysis.ipynb                # Exploratory data analysis
│   ├── 03_data_stratification_encoding.ipynb   # Data preparation for ML
│   ├── 04_model_random_forest.ipynb            # Random Forest model
│   ├── 05_model_multi_logistic_regression.ipynb # Logistic Regression model
│   ├── 06_model_knn.ipynb                      # K-Nearest Neighbors model
│   ├── 07_model_lightGBM.ipynb                 # LightGBM model
│   ├── 08_model_SVM.ipynb                      # Support Vector Machine model
│   ├── 09.model_neural_networks.ipynb          # Neural Network model
│   └── X1_model_comparison_analysis.ipynb      # Comprehensive model comparison
│
└── Visualizations/                     # 📈 Generated plots and charts
    ├── Report/                         # 📁 Dissertation report visualizations
    │   ├── 2_1_database_scheme.png            # Database schema diagram
    │   ├── 3_1_project_pipeline.png           # Overall project pipeline
    │   ├── 3_2_features_pipeline.png          # Feature engineering pipeline
    │   ├── 4_1_data_class_distribution.png    # Target class distribution analysis
    │   ├── 4_2_weighted_data_class.png        # Class weighting strategy
    │   ├── 4_3_model_performance_comparison.png # Model performance comparison
    │   ├── 4_4_model_training_time.png        # Model training time analysis
    │   ├── 4_5_model_runtime_gridSearchCV.png # Hyperparameter tuning runtime
    │   ├── 4_6_radar_plot_comparison.png      # Radar plot model comparison
    │   ├── 4_7_feature_importance.png         # Feature importance analysis
    │   ├── A1_Gender_plot.png                 # Gender distribution analysis
    │   ├── A1_Age_band_plot.png               # Age distribution analysis
    │   ├── A1_HE_plot.png                     # Higher education background
    │   ├── A1_Disability_plot.png             # Disability status analysis
    │   ├── A1_IMD_band_plot.png               # IMD band distribution
    │   └── A1_Region_plot.png                 # Regional distribution analysis
    │
    └── Model_comparison/               # 📁 Model comparison visualizations
        ├── 01_class_distribution.png          # Target class distribution
        ├── 02_model_comparison.png            # Model performance comparison
        ├── 03_class_level_performance.png     # Per-class performance metrics
        └── 04_confusion_matrices.png          # Confusion matrices grid
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter lightgbm
```

### Running the Analysis
1. **Data Setup**: Place CSV files in the `Data/` folder
2. **Complete Pipeline**: Open `Notebooks/01_data_ingest_cleaning.ipynb` and run sequentially through `X1_model_comparison_analysis.ipynb`
3. **Individual Models**: Run specific model notebooks (04-09) for focused analysis
4. **Custom Analysis**: Import modules from `Python_files/` for programmatic use

## 📊 Models & Results

### Implemented Models
- **Random Forest** - Ensemble tree-based classifier
- **Logistic Regression** - Linear classification with regularization
- **K-Nearest Neighbors** - Instance-based learning
- **LightGBM** - Gradient boosting framework
- **Support Vector Machine** - Kernel-based classification
- **Neural Networks** - Multi-layer perceptron

### Key Features
- **Feature Engineering**: VLE engagement metrics, demographic encoding
- **Model Optimization**: Hyperparameter tuning with cross-validation
- **Performance Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Feature Importance**: Analysis of predictive factors across models
- **Comprehensive Comparison**: Head-to-head model performance analysis

## 📈 Outputs

### Model Artifacts
- Trained models saved as `.pkl` files
- Feature scalers for models requiring normalization
- Performance metrics in JSON format
- Classification reports with detailed per-class statistics

### Visualizations
- Model performance comparison charts
- Confusion matrices for all models
- Feature importance rankings
- Class distribution analysis

### Data Products
- Processed and encoded datasets
- Train/test splits ready for ML workflows
- Feature engineering pipeline outputs

## 🔧 Customization

The modular design allows for easy modification:
- **Add new models**: Create new notebook following the established pattern
- **Modify features**: Update `encoding_utils.py` and `data_processor.py`
- **Change evaluation metrics**: Modify `analysis_engine.py`
- **Custom visualizations**: Extend `visualization.py`

## 📋 Workflow

1. **Data Ingestion** → Clean and validate raw CSV files
2. **Exploration** → Visual analysis and feature understanding
3. **Preprocessing** → Stratification, encoding, and train/test splits
4. **Modeling** → Train and optimize individual models
5. **Evaluation** → Compare performance across all models
6. **Deployment** → Saved models ready for production use

---

**🎯 Goal**: Predict student academic outcomes (Withdrawn/Fail/Pass/Distinction) using demographic and engagement data through optimized machine learning models.

## 📋 Conclusion

This project addresses the critical challenge of high attrition rates in virtual learning environments through a comprehensive machine learning pipeline designed to identify at-risk students early and enable timely interventions. Using the Open University Learning Analytics Dataset, the analysis classifies student outcomes into three categories: Withdrawn (class 0), Fail (class 1), and Pass (class 2).

### Key Approach
- **Engagement Indicators**: Created three interpretable engagement metrics: excellent score (≥70), VLE activity, and composite student engagement flags
- **Contextual Factors**: Incorporated socioeconomic and demographic variables (highest education, IMD band, disability, age, region)
- **Balanced Methodology**: Employed cohort and outcome stratified data splitting to maintain class proportions
- **Comprehensive Model Evaluation**: Assessed six machine learning algorithms with custom class weights and dropout-focused metrics

### Results
The Multinomial Logistic Regression model demonstrated exceptional performance for early-warning applications with:
- **66.8% recall** for identifying Withdrawn students
- **Efficient training time** (~4 minutes)
- **Interpretable coefficients** for actionable insights
- **Optimized hyperparameters** through stratified 5-fold GridSearchCV

### Impact
This research contributes to educational data mining by providing a robust, interpretable solution for early student risk identification, supporting institutional efforts to improve retention and student success through data-driven interventions.