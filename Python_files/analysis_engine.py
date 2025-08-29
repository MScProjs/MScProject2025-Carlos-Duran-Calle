"""
Statistical analysis module for Student Assessment Analysis
Contains functions for cross-tabulation analysis and regression calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from config_file import CATEGORICAL_VARS, VARIABLE_CONFIGS, RESULT_CODE_MAPPING


def create_cross_tables(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create cross-tabulation tables for all categorical variables.
    
    Args:
        data: Processed student data DataFrame
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cross-tabulation tables
    """
    print("Creating cross-tabulation tables...")
    
    cross_tables = {}
    
    for var in CATEGORICAL_VARS:
        if var in data.columns:
            print(f"  Creating cross table for: {var}")
            
            # Create cross table with proportions
            ctab = pd.crosstab(
                data[var],
                data['final_result_code'],
                normalize='index'
            ).round(3)
            
            cross_tables[var] = ctab
            
            # Print summary
            print(f"    Categories: {len(ctab)}, Outcomes: {len(ctab.columns)}")
            
        else:
            print(f"  ⚠ Variable {var} not found in data")
    
    print(f"✓ Created {len(cross_tables)} cross-tabulation tables")
    return cross_tables


def calculate_linear_regression(x: list, y: list) -> Dict:
    """
    Calculate linear regression and return slope, intercept, and R².
    
    Args:
        x: Independent variable values
        y: Dependent variable values
        
    Returns:
        Dict: Regression statistics
    """
    if len(x) != len(y) or len(x) < 2:
        return {
            'slope': 0,
            'intercept': 0,
            'r_squared': 0,
            'y_pred': [0] * len(y)
        }
    
    try:
        X = np.array(x).reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'slope': reg.coef_[0],
            'intercept': reg.intercept_,
            'r_squared': r2,
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"Warning: Regression calculation failed: {e}")
        return {
            'slope': 0,
            'intercept': 0,
            'r_squared': 0,
            'y_pred': [0] * len(y)
        }


def analyze_cross_table_trends(cross_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze trends in cross-tabulation tables using linear regression.
    
    Args:
        cross_tables: Dictionary of cross-tabulation tables
        
    Returns:
        pd.DataFrame: Summary of regression analysis
    """
    print("Analyzing cross-table trends...")
    
    summary_data = []
    
    for var_name, cross_table in cross_tables.items():
        if var_name in VARIABLE_CONFIGS:
            config = VARIABLE_CONFIGS[var_name]
            x_values = config['x_values']
            
            # Handle special cases
            if var_name == 'region':
                # Sort regions alphabetically and create numeric values
                sorted_regions = sorted(cross_table.index.tolist())
                cross_table = cross_table.reindex(sorted_regions)
                x_values = list(range(len(sorted_regions)))
                
            elif var_name == 'highest_education':
                # Reorder cross table according to education hierarchy
                if 'order' in config:
                    cross_table = cross_table.reindex(config['order'])
            
            # Analyze each outcome
            for col in cross_table.columns:
                y_values = cross_table[col].values
                regression = calculate_linear_regression(x_values, y_values)
                
                # Determine trend strength
                r2 = regression['r_squared']
                if r2 > 0.7:
                    strength = 'Strong'
                elif r2 > 0.3:
                    strength = 'Moderate'
                else:
                    strength = 'Weak'
                
                summary_data.append({
                    'Variable': var_name.replace('_', ' ').title(),
                    'Outcome': RESULT_CODE_MAPPING.get(col, f'Code {col}'),
                    'R_Squared': regression['r_squared'],
                    'Slope': regression['slope'],
                    'Intercept': regression['intercept'],
                    'Trend': 'Positive' if regression['slope'] > 0 else 'Negative',
                    'Strength': strength
                })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"✓ Analyzed trends for {len(summary_df)} variable-outcome combinations")
    
    return summary_df


def get_regression_insights(summary_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate insights from regression analysis.
    
    Args:
        summary_df: Summary DataFrame from regression analysis
        
    Returns:
        Dict[str, str]: Key insights
    """
    insights = {}
    
    if summary_df.empty:
        insights['no_data'] = "No regression data available for analysis"
        return insights
    
    # Find strongest correlations
    strongest = summary_df.loc[summary_df['R_Squared'].idxmax()]
    insights['strongest_correlation'] = (
        f"Strongest correlation: {strongest['Variable']} with {strongest['Outcome']} "
        f"(R² = {strongest['R_Squared']:.3f}, {strongest['Trend']} trend)"
    )
    
    # Find variables with consistent trends across outcomes
    for var in summary_df['Variable'].unique():
        var_data = summary_df[summary_df['Variable'] == var]
        trends = var_data['Trend'].unique()
        if len(trends) == 1:
            trend = trends[0]
            avg_r2 = var_data['R_Squared'].mean()
            insights[f'{var.lower().replace(" ", "_")}_consistent'] = (
                f"{var} shows consistent {trend.lower()} trends across all outcomes "
                f"(avg R² = {avg_r2:.3f})"
            )
    
    # Find high R² relationships
    high_r2 = summary_df[summary_df['R_Squared'] > 0.5]
    if not high_r2.empty:
        insights['high_r2_relationships'] = (
            f"Strong relationships (R² > 0.5): "
            f"{len(high_r2)} out of {len(summary_df)} combinations"
        )
    
    # Find the variable with highest average correlation
    avg_r2_by_var = summary_df.groupby('Variable')['R_Squared'].mean().sort_values(ascending=False)
    if not avg_r2_by_var.empty:
        best_var = avg_r2_by_var.index[0]
        best_r2 = avg_r2_by_var.iloc[0]
        insights['best_predictor'] = (
            f"Best predictor variable: {best_var} (avg R² = {best_r2:.3f})"
        )
    
    return insights


def analyze_student_segments(data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze different student segments based on engagement and performance.
    
    Args:
        data: Processed student data
        
    Returns:
        Dict[str, Dict]: Analysis results for different segments
    """
    print("Analyzing student segments...")
    
    segments = {}
    total_students = len(data)
    
    # High performers (excellent score)
    high_performers = data[data['excellent_Score'] == 1]
    segments['high_performers'] = {
        'count': len(high_performers),
        'percentage': len(high_performers) / total_students * 100,
        'avg_vle_clicks': high_performers['total_click_vle'].mean(),
        'completion_rate': (high_performers['final_result_code'] >= 2).mean() * 100
    }
    
    # High VLE engagement
    high_engagement = data[data['active_in_VLE'] == 1]
    segments['high_engagement'] = {
        'count': len(high_engagement),
        'percentage': len(high_engagement) / total_students * 100,
        'avg_score': high_engagement['score'].mean(),
        'completion_rate': (high_engagement['final_result_code'] >= 2).mean() * 100
    }
    
    # Overall engaged students (either high score or high VLE)
    overall_engaged = data[data['student_engagementt'] == 1]
    segments['overall_engaged'] = {
        'count': len(overall_engaged),
        'percentage': len(overall_engaged) / total_students * 100,
        'completion_rate': (overall_engaged['final_result_code'] >= 2).mean() * 100,
        'avg_score': overall_engaged['score'].mean()
    }
    
    # Students with no VLE activity
    no_vle = data[data['total_click_vle'] == 0]
    segments['no_vle'] = {
        'count': len(no_vle),
        'percentage': len(no_vle) / total_students * 100,
        'avg_score': no_vle['score'].mean(),
        'withdrawal_rate': (no_vle['final_result_code'] == 0).mean() * 100
    }
    
    # Low performers (bottom 25% of scores)
    score_25th = data['score'].quantile(0.25)
    low_performers = data[data['score'] <= score_25th]
    segments['low_performers'] = {
        'count': len(low_performers),
        'percentage': len(low_performers) / total_students * 100,
        'avg_vle_clicks': low_performers['total_click_vle'].mean(),
        'withdrawal_rate': (low_performers['final_result_code'] == 0).mean() * 100
    }
    
    print(f"✓ Analyzed {len(segments)} student segments")
    return segments


def run_complete_analysis(data: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict, Dict]:
    """
    Run the complete statistical analysis pipeline.
    
    Args:
        data: Processed student data
        
    Returns:
        Tuple containing:
        - cross_tables: Cross-tabulation tables
        - summary_stats: Regression summary statistics
        - insights: Key insights from analysis
        - segments: Student segment analysis
    """
    print("\n" + "="*60)
    print("STARTING STATISTICAL ANALYSIS")
    print("="*60)
    
    try:
        # Create cross tables
        cross_tables = create_cross_tables(data)
        
        # Analyze trends
        summary_stats = analyze_cross_table_trends(cross_tables)
        
        # Generate insights
        insights = get_regression_insights(summary_stats)
        
        # Analyze student segments
        segments = analyze_student_segments(data)
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS COMPLETED")
        print("="*60)
        
        # Print key findings
        print("\nKEY INSIGHTS:")
        for key, insight in insights.items():
            print(f"  • {insight}")
        
        print(f"\nSTUDENT SEGMENTS:")
        for segment_name, stats in segments.items():
            name = segment_name.replace('_', ' ').title()
            print(f"  • {name}: {stats['count']:,} students ({stats['percentage']:.1f}%)")
        
        return cross_tables, summary_stats, insights, segments
        
    except Exception as e:
        print(f"\n✗ Error during statistical analysis: {e}")
        raise


if __name__ == "__main__":
    print("Analysis engine module ready for import")
    print("Use run_complete_analysis(processed_data) to run the complete analysis")