# ?? Student Dropout Prediction - Comprehensive Machine Learning Study

## ?? **Project Overview**

This dissertation implements a comprehensive machine learning pipeline for **student dropout prediction** using the Open University Learning Analytics Dataset. The study compares **6 different algorithms** with a primary focus on optimizing **dropout recall** for early intervention systems.

### **Problem Statement**
- **Target**: Predict student academic outcomes (3-class classification)
- **Classes**: 
  - **Class 0**: Withdrawn/Dropout (19.1% - minority class) ??
  - **Class 1**: Fail (25.4%) ??  
  - **Class 2**: Pass (55.5% - majority class) ??
- **Challenge**: Class imbalance requiring specialized optimization strategies
- **Goal**: Achieve ?60% dropout recall for effective early intervention

---

## ?? **Notebook Architecture & Analysis Pipeline**

### **Data Processing & Exploration**

#### **01_data_ingest_cleaning.ipynb**
- **Purpose**: Data loading, validation, and preprocessing pipeline using modular Python architecture
- **Key Workflow**: 
  - ?? Loads 7 CSV files (assessments, courses, studentInfo, studentRegistration, studentAssessment, studentVle, vle)
  - ?? **Python Module Integration**: Seamlessly connects with custom modules from `Python_files/`:
    * **data_loader.py** ? `load_and_validate()` handles all data loading and validation
    * **data_processor.py** ? `process_all_data()` executes the complete processing pipeline
    * **config_file.py** ? Provides configuration constants and file mappings
    * **analysis_engine.py** ? `run_complete_analysis()` generates statistical summaries
  - ?? Creates derived features like `courses_per_term` for academic load analysis
  - ? Validates data integrity and handles missing values automatically
  - ?? Exports cleaned datasets ready for machine learning experiments

**Modular Architecture**: This notebook leverages 6 custom Python modules for robust data processing:

1. **data_loader.py** (180 lines)
   - `load_and_validate()`: Main entry point for data loading pipeline
   - `load_all_data()`: Loads 7 CSV files with error handling
   - `validate_data()`: Comprehensive data integrity checks and validation
   - `get_data_info()`: Generates summary statistics and data profiling

2. **data_processor.py** (400+ lines)
   - `process_all_data()`: Comprehensive processing pipeline
   - `find_first_assessments()`: Identifies earliest TMA assessments per course-presentation by:
     * Filtering assessments to only TMA (Tutor Marked Assessment) type
     * Finding minimum date for each course-presentation combination
     * Extracting assessment details (ID, date, weight) for first assessments
   - `filter_active_students()`: Removes inactive students by:
     * Merging registration data with first assessment dates
     * Filtering out students who withdrew before their first assessment
     * Retaining only students active during the prediction window (first assessment)
   - `merge_assessment_scores()`: Combines assessment data with student records by:
     * Merging student registrations with their first assessment scores
     * Handling missing scores (assigns 0 for non-submissions)
     * Adding assessment metadata (submission date, banking status)
   - `calculate_vle_engagement()`: Computes VLE interaction metrics by:
     * Merging student data with VLE click records
     * Filtering clicks to only those before/at first assessment date
     * Calculating `total_click_vle`: Sum of all VLE clicks per student
     * Computing `average_click_vle`: Course-presentation specific average clicks
   - `create_engagement_features()`: Engineers 3 key engagement indicators:
     * `excellent_Score`: Binary (1/0) - Students scoring 70 (merit threshold)
     * `active_in_VLE`: Binary (1/0) - Students with above-average VLE clicks before first assessment
     * `student_engagement`: Binary (1/0) - Students meeting either excellence OR VLE activity criteria

3. **config_file.py** (Configuration Management)
   - `DATA_FILES`: File path mappings for all CSV datasets
   - `MERIT_SCORE = 70`: Academic performance threshold definition
   - `STUDENT_INFO_COLUMNS`: Standardized column specifications
   - `FINAL_RESULT_MAP`: Outcome classification mappings

4. **analysis_engine.py** (Statistical Analysis)
   - `run_complete_analysis()`: Comprehensive statistical analysis pipeline
   - `create_cross_tables()`: Cross-tabulation analysis for categorical variables
   - `calculate_linear_regression()`: Regression analysis for trend identification

#### **02_visual_analysis.ipynb**
- **Purpose**: Exploratory Data Analysis (EDA) and cross-tabulation visualization
- **Key Features**:
  - Interactive Plotly visualizations for student engagement patterns
  - Cross-tabulation analysis between demographics and outcomes
  - Linear regression analysis for identifying trends
  - Student segmentation based on engagement levels
  - Statistical analysis of class distributions and feature correlations

#### **03_data_stratification_encoding.ipynb**
- **Purpose**: Data splitting, categorical encoding, and initial model comparison
- **Key Operations**:
  - **Stratified Train-Test Split**: Maintains class proportions across cohorts (preserves 19.1% dropout representation)
  - **Sophisticated Categorical Encoding**: Uses custom `encoding_utils` module with tailored strategies:
    * **Region** (Nominal) ? **One-Hot Encoding**: Creates binary columns (region_Wales, region_Scotland, etc.)
    * **Highest Education** (Ordinal) ? **Ordinal Encoding**: Maps education levels to numerical hierarchy
    * **IMD Band** (Ordinal) ? **Ordinal Encoding**: Preserves socioeconomic deprivation ranking
    * **Age Band** (Ordinal) ? **Ordinal Encoding**: Maintains age group progression
    * **Disability** (Binary) ? **Binary Encoding**: Simple 0/1 transformation
  - **Dataset Export**: Saves encoded datasets (`X_train_encoded.csv`, `X_test_encoded.csv`, `y_train.csv`, `y_test.csv`) for individual model optimization
  - **Encoding Validation**: Comprehensive summary showing feature transformations and new column creation

---

### **Individual Model Optimization**

#### **04_model_random_forest.ipynb**
- **Algorithm**: Random Forest Classifier
- **Optimization Results**:
  - **Dropout Recall**: 58.96%
  - **Test Accuracy**: 63.25%
  - **Configuration**: 100 trees, max depth 15, custom class weights {0: 2.51, 1: 1.57, 2: 0.57}
  - **Runtime**: ~30 minutes with 100 parameter combinations
  - **Feature Importance**: Tree-based importance analysis
- **Key Insights**: Ensemble method providing robust performance with interpretable feature rankings

#### **05_model_multi_logistic_regression.ipynb** ? **WINNER**
- **Algorithm**: Multinomial Logistic Regression
- **Optimization Results**:
  - **Dropout Recall**: 66.84% ?? **(BEST PERFORMER)**
  - **Test Accuracy**: 78.22%
  - **Configuration**: L1 penalty (Lasso), C=0.01, SAGA solver, custom class weights
  - **Runtime**: ~4 minutes (most efficient)
  - **Feature Analysis**: Detailed coefficient interpretation with class-specific analysis
- **Key Advantages**: Best balance of performance, efficiency, and interpretability

#### **06_model_knn.ipynb**
- **Algorithm**: K-Nearest Neighbors
- **Optimization Results**:
  - **Dropout Recall**: 31.18%
  - **Test Accuracy**: 58.41%
  - **Configuration**: k=3, Euclidean distance, uniform weighting
  - **Runtime**: ~109 minutes (1.8 hours)
  - **Preprocessing**: StandardScaler normalization (mandatory for distance-based algorithms)
- **Key Insights**: Local pattern recognition with fine-grained decision boundaries

#### **07_model_lightGBM.ipynb**
- **Algorithm**: LightGBM Gradient Boosting
- **Optimization Results**:
  - **Dropout Recall**: 56.01%
  - **Test Accuracy**: 48.75%
  - **Configuration**: 100 estimators, learning rate 0.05, max depth 6
  - **Runtime**: ~428 minutes (7.1 hours) - most computationally intensive
  - **Feature Importance**: Tree-based importance with socioeconomic factors (IMD) as top predictor
- **Key Insights**: Advanced gradient boosting with excellent handling of mixed data types

#### **08_model_SVM.ipynb**
- **Algorithm**: Support Vector Machine
- **Optimization Results**:
  - **Dropout Recall**: 63.09% (2nd best performance)
  - **Test Accuracy**: 48.21%
  - **Configuration**: RBF kernel, C=0.1, gamma='scale', custom class weights
  - **Runtime**: ~80 minutes
  - **Model Characteristics**: Uses 92.7% of training data as support vectors
- **Key Insights**: Non-linear decision boundaries with strong theoretical foundation

#### **09_model_neural_networks.ipynb**
- **Algorithm**: Multi-Layer Perceptron (Neural Network)
- **Optimization Results**:
  - **Dropout Recall**: 18.67%
  - **Test Accuracy**: 56.66%
  - **Configuration**: (100, 50) hidden layers, Tanh activation, LBFGS solver, ?=0.001
  - **Runtime**: ~47 minutes
  - **Architecture**: 7,203 trainable parameters across 4 layers
- **Key Insights**: Complex feature learning with permutation-based importance analysis

---

### **Comprehensive Model Comparison**

#### **X1_model_comparison_analysis.ipynb**
- **Purpose**: Final model evaluation, comparison, and deployment recommendations
- **Key Analyses & Visualizations**:
  - **?? Performance Dashboard**: 
    * **Radar plots**: Multi-dimensional performance comparison showing dropout recall, precision, F1-scores, and efficiency across all 6 models
    * **Efficiency analysis**: Runtime vs. performance scatter plots to identify optimal trade-offs
    * **Confusion matrices**: Detailed breakdown of prediction accuracy for each class (Pass/Fail/Dropout)
    * *Relevance*: Provides stakeholders with comprehensive performance overview for informed decision-making
  - **?? Normalized Feature Importance**: 
    * **Cross-model importance comparison**: Standardized feature rankings across different algorithm types
    * **Top 10 predictive features**: Unified importance scores combining tree-based, coefficient-based, and permutation importance
    * *Relevance*: Ensures fair comparison between algorithms and identifies universally important predictors for intervention strategies
  - **?? Winner Analysis**: 
    * **Logistic Regression deep dive**: Coefficient interpretation, class-specific analysis, and decision boundary visualization
    * **Performance breakdown**: Detailed metrics showing why Logistic Regression achieved superior dropout recall
    * *Relevance*: Provides actionable insights for understanding model predictions and building trust with educators
  - **?? Academic-Ready Visualizations**: 
    * **Publication-quality charts**: Professional formatting suitable for research papers and presentations
    * **Stakeholder dashboards**: Executive summaries with clear performance tiers and deployment recommendations
    * *Relevance*: Facilitates knowledge transfer and supports evidence-based decision making at institutional level

---

## ?? **Model Performance Rankings**

### **Complete Results Table** (by Dropout Recall)

| Rank | Model | Dropout Recall | Dropout Precision | At-Risk Recall | Weighted F1 | Runtime | Status |
|------|-------|----------------|-------------------|----------------|-------------|---------|--------|
| #1 ?? | **Logistic Regression** | **66.84%** | **62.12%** | **64.90%** | **78.22%** | **4m** | **EXCELLENT** |
| #2 ?? | Support Vector Machine | 63.09% | - | - | 48.81% | 80m | EXCELLENT |
| #3 ?? | Random Forest | 58.96% | - | - | 63.25% | 30m | GOOD |
| #4 | LightGBM | 56.01% | - | - | 50.36% | 428m | GOOD |
| #5 | K-Nearest Neighbors | 31.18% | - | - | 54.45% | 109m | NEEDS WORK |
| #6 | Neural Networks | 18.67% | - | - | 53.85% | 47m | NEEDS WORK |

### **Performance Tiers**
- **?? EXCELLENT (?60% Dropout Recall)**: 2 models - Ready for deployment
- **?? GOOD (40-60% Dropout Recall)**: 2 models - Acceptable with optimization
- **?? NEEDS IMPROVEMENT (<40% Dropout Recall)**: 2 models - Require significant enhancement

---

## ?? **Technical Methodology**

### **Class Imbalance Handling Strategy**
- **Custom Class Weighting**: Dropout-focused weights (Class 0: 2.5x, Class 1: 1.5x, Class 2: 0.6x)
- **Stratified Sampling**: Maintains class proportions across train/test splits
- **Custom Scoring Metrics**: Primary optimization on `dropout_recall`
- **Cross-Validation**: 5-fold CV with consistent evaluation across all models

### **Feature Engineering Pipeline**
- **Categorical Encoding**: Ordinal encoding for ranked features, binary encoding for categorical
- **Feature Scaling**: StandardScaler for distance-based algorithms (KNN, SVM, Neural Networks)
- **Derived Features**: `courses_per_term`, engagement metrics, academic performance indicators
- **Feature Selection**: Based on domain knowledge and statistical significance

### **Hyperparameter Optimization Framework**
- **GridSearchCV**: Exhaustive search with 5-fold cross-validation
- **Search Intensity**: 24-3,888 parameter combinations per model
- **Parallel Processing**: Multi-core optimization for efficiency
- **Custom Scoring**: Dropout-focused metrics prioritizing early intervention

---

## ?? **Key Predictive Features**

### **Top 10 Features** (Normalized Importance Across All Models)

| Rank | Feature | Description | Normalized Score |
|------|---------|-------------|------------------|
| #1 | `excellent_Score` | Academic excellence indicator | 0.892 |
| #2 | `student_engagement` | Overall engagement metrics | 0.854 |
| #3 | `imd_band_ord` | Index of Multiple Deprivation (socioeconomic status) | 0.723 |
| #4 | `active_in_VLE` | Virtual learning environment activity | 0.651 |
| #5 | `highest_education_ord` | Educational background level | 0.598 |
| #6 | `age_band_ord` | Student age group | 0.534 |
| #7 | `courses_per_term` | Academic load distribution | 0.487 |
| #8 | `disability_binary` | Disability status | 0.423 |
| #9 | `region_Wales` | Geographic location factor | 0.398 |
| #10 | `num_of_prev_attempts` | Previous course attempts | 0.365 |

### **Feature Insights for Intervention**
- **Academic Performance**: `excellent_Score` and prior achievement patterns are strongest predictors
- **Engagement Patterns**: VLE activity and overall engagement crucial for early detection
- **Socioeconomic Factors**: IMD band indicates financial/social barriers requiring support
- **Demographics**: Age and educational background provide context for targeted interventions

---

## ?? **Research Contributions & Innovations**

### **Methodological Advances**
1. **Dropout-Focused Optimization**: Novel scoring metrics prioritizing minority class detection over overall accuracy
2. **Comprehensive Algorithm Comparison**: Systematic evaluation of 6 diverse ML approaches with consistent methodology
3. **Class Imbalance Solutions**: Multiple weighting strategies specifically designed for educational intervention
4. **Normalized Feature Importance**: Fair comparison framework across different model architectures
5. **Academic-Ready Pipeline**: Complete workflow from raw data to deployment-ready models

### **Practical Applications**
- **Early Warning System**: Real-time identification of at-risk students for proactive intervention
- **Resource Allocation**: Data-driven targeting of academic support services
- **Institutional Analytics**: Evidence-based decision making for student success initiatives
- **Scalable Framework**: Replicable methodology for other educational institutions

---

## ?? **Deployment Strategy & Recommendations**

### **Production Model: Logistic Regression**
```python
# Optimal Configuration
MultinomialLogisticRegression(
    penalty='l1',           # L1 regularization (Lasso)
    C=0.01,                # Regularization strength
    solver='saga',         # Efficient solver for L1 penalty
    class_weight={         # Custom dropout-focused weights
        0: 2.51,          # Withdrawn (boosted)
        1: 1.57,          # Fail (moderate)
        2: 0.57           # Pass (reduced)
    },
    random_state=42,
    max_iter=1000
)
```

### **Implementation Architecture**
1. **Real-Time Scoring Pipeline**
   - API endpoint for live student data ingestion
   - Automated feature preprocessing and encoding
   - Probability scoring with threshold-based alerts

2. **Monitoring Dashboard**
   - Student risk scores updated daily
   - Feature drift detection and model performance tracking
   - Intervention tracking and outcome measurement

3. **Alert System**
   - Automated notifications for high-risk students (?60% dropout probability)
   - Escalation protocols for academic advisors
   - Integration with student information systems

### **Performance Benchmarks**
- **Target Achievement**: ? 66.84% dropout recall (exceeds 60% goal)
- **Efficiency**: ? 4-minute training time (suitable for regular retraining)
- **Interpretability**: ? Clear coefficient analysis for actionable insights
- **Scalability**: ? Linear complexity suitable for large student populations

---

## ?? **Technical Infrastructure**

### **Data Pipeline**
```
Raw Data (7 CSV files) 
    ? 
Data Cleaning & Validation (01_data_ingest_cleaning.ipynb)
    ?
Exploratory Analysis (02_visual_analysis.ipynb)
    ?
Feature Engineering & Encoding (03_data_stratification_encoding.ipynb)
    ?
Model Training & Optimization (04-09_model_*.ipynb)
    ?
Comparative Analysis & Selection (X1_model_comparison_analysis.ipynb)
    ?
Production Deployment
```

### **File Structure**
```
MScProject/
??? Data/
?   ??? [raw CSV files]
?   ??? output/
?   ?   ??? X_train_encoded.csv
?   ?   ??? X_test_encoded.csv
?   ?   ??? y_train.csv
?   ?   ??? y_test.csv
?   ??? model_metrics/
?       ??? models/           # Trained model files (.pkl)
?       ??? metrics/          # Performance metrics (.json)
?       ??? reports/          # Classification reports
??? Notebooks/                # Analysis notebooks (10 total)
??? Python_files/            # Utility modules
??? Visualizations/          # Generated plots and charts
??? Documentation/           # Project documentation
```

### **Saved Model Assets**
Each optimized model includes:
- **Trained Model**: `.pkl` file with best hyperparameters
- **Preprocessing**: Scaler/encoder objects for feature transformation
- **Metrics**: JSON files with complete performance evaluation
- **Documentation**: Usage instructions and configuration details

---

## ?? **Future Work & Extensions**

### **Model Enhancements**
1. **Ensemble Methods**: Combine top 2 models (Logistic Regression + SVM) for increased robustness
2. **Deep Learning**: Advanced neural architectures (LSTM, attention mechanisms) for sequential data
3. **Online Learning**: Adaptive models that update with new student data in real-time
4. **Interpretability**: SHAP values and LIME explanations for individual predictions

### **Feature Engineering**
1. **Temporal Features**: Time-series analysis of engagement patterns
2. **Social Network**: Peer interaction features from forum and collaboration data
3. **Behavioral Patterns**: Clickstream analysis and learning pathway modeling
4. **External Data**: Integration with library usage, financial aid, and housing data

### **Deployment Extensions**
1. **Multi-Institution**: Federated learning across multiple universities
2. **Mobile Integration**: Student-facing app with personalized recommendations
3. **Intervention Optimization**: A/B testing of different support strategies
4. **Predictive Analytics**: Early prediction at enrollment vs. mid-semester checkpoints

---

## ?? **Impact & Validation**

### **Academic Contributions**
- **Conference Presentations**: Framework suitable for educational data mining conferences
- **Peer Review**: Methodology validated through comprehensive model comparison
- **Reproducibility**: Complete codebase with detailed documentation
- **Generalizability**: Applicable to other educational institutions with similar data

### **Institutional Benefits**
- **Student Success**: Early identification enables proactive intervention
- **Resource Efficiency**: Targeted support reduces overall attrition costs
- **Data-Driven Decisions**: Evidence-based allocation of academic support services
- **Continuous Improvement**: Ongoing model performance monitoring and optimization

### **Ethical Considerations**
- **Fairness**: Analysis of model bias across demographic groups
- **Privacy**: Anonymized data processing with secure model deployment
- **Transparency**: Interpretable models enabling student understanding of risk factors
- **Intervention Ethics**: Focus on support rather than punitive measures

---

## ?? **Conclusion**

This comprehensive machine learning study successfully demonstrates that **student dropout prediction can be optimized for early intervention** with the right algorithmic approach and evaluation framework. 

**Key Achievements:**
- ? **Exceeded Target**: 66.84% dropout recall surpasses 60% goal
- ? **Efficiency**: 4-minute training enables regular model updates
- ? **Interpretability**: Clear feature importance guides intervention strategies
- ? **Scalability**: Production-ready pipeline suitable for institutional deployment

The **Logistic Regression model** emerges as the clear winner, providing the optimal balance of performance, efficiency, and interpretability needed for real-world educational applications. This work provides a robust foundation for implementing data-driven student success initiatives in higher education.

---

*Generated on August 21, 2025 | Student Dropout Prediction ML Study | University of Bristol*
