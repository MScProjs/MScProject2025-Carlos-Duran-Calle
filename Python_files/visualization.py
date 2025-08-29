"""
Visualization module for Student Assessment Analysis
Contains functions for creating interactive plots and charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple
from analysis_engine import calculate_linear_regression
from config_file import VARIABLE_CONFIGS, RESULT_CODE_MAPPING, PLOT_COLORS, PLOT_STYLE


def setup_matplotlib_style():
    """Setup matplotlib style for consistent plotting."""
    try:
        plt.style.use(PLOT_STYLE)
        sns.set_palette("husl")
        print("✓ Matplotlib style configured")
    except Exception as e:
        print(f"⚠ Could not set matplotlib style: {e}")
        # Use default style
        pass


def create_plotly_regression_plot(cross_table: pd.DataFrame, 
                                variable_name: str, 
                                x_values: List[float], 
                                x_label: str, 
                                categories: List[str] = None) -> go.Figure:
    """
    Create an interactive regression plot using Plotly for a single variable.
    
    Args:
        cross_table: DataFrame with categories as index and result codes as columns
        variable_name: Name of the variable for the title
        x_values: Numeric values for x-axis
        x_label: Label for x-axis
        categories: Category names (if different from cross_table.index)
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    if categories is None:
        categories = cross_table.index.tolist()
    
    # Process each result code column
    for i, col in enumerate(cross_table.columns):
        y_values = cross_table[col].values
        color = PLOT_COLORS[i] if i < len(PLOT_COLORS) else f'hsl({i*60}, 70%, 50%)'
        result_name = RESULT_CODE_MAPPING.get(col, f'Code {col}')
        
        # Calculate regression
        regression = calculate_linear_regression(x_values, y_values)
        
        # Format slope for display
        slope_text = f"slope = {regression['slope']:.4f}"
        
        # Add scatter plot (data points)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name=f'{result_name} data',
            marker=dict(color=color, size=12, opacity=0.8),
            text=categories,
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_label}: %{{x}}<br>' +
                         'Proportion: %{y:.3f}<br>' +
                         f'Result: {result_name}<extra></extra>'
        ))
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=regression['y_pred'],
            mode='lines',
            name=f'{result_name} fit (R² = {regression["r_squared"]:.3f}, {slope_text})',
            line=dict(color=color, width=3),
            hovertemplate=f'{result_name} Trend Line<br>' +
                         f'R² = {regression["r_squared"]:.3f}<br>' +
                         f'Slope = {regression["slope"]:.4f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Linear Regression: Student Outcomes - {variable_name}',
            font=dict(size=16, color='#333')
        ),
        xaxis=dict(
            title=x_label,
            gridcolor='#e0e0e0'
        ),
        yaxis=dict(
            title='Proportion',
            gridcolor='#e0e0e0',
            range=[0, max(cross_table.values.flatten()) * 1.1] if cross_table.values.size > 0 else [0, 1]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ddd',
            borderwidth=1,
            font=dict(size=10)
        ),
        height=600,
        width=900,
        margin=dict(t=60, r=50, b=80, l=80)
    )
    
    return fig


def plot_all_cross_tables(cross_tables: Dict[str, pd.DataFrame]) -> Dict[str, go.Figure]:
    """
    Create and display all cross table plots individually.
    
    Args:
        cross_tables: Dictionary with variable names as keys and cross tables as values
        
    Returns:
        Dict[str, go.Figure]: Dictionary of Plotly figures
    """
    figures = {}
    
    print("Creating individual regression plots for each categorical variable...\n")
    
    for var_name, cross_table in cross_tables.items():
        if var_name in VARIABLE_CONFIGS:
            config = VARIABLE_CONFIGS[var_name].copy()
            
            print(f"📊 Creating plot for {var_name.replace('_', ' ').title()}")
            
            # Handle special cases
            if var_name == 'region':
                # Sort regions alphabetically
                sorted_regions = sorted(cross_table.index.tolist())
                cross_table_sorted = cross_table.reindex(sorted_regions)
                config['x_values'] = list(range(len(sorted_regions)))
                config['categories'] = sorted_regions
                cross_table = cross_table_sorted
                
            elif var_name == 'highest_education':
                # Order by education level
                if 'order' in config:
                    cross_table = cross_table.reindex(config['order'])
            
            # Create the plot
            try:
                fig = create_plotly_regression_plot(
                    cross_table,
                    var_name.replace('_', ' ').title(),
                    config['x_values'],
                    config['x_label'],
                    config['categories']
                )
                
                figures[var_name] = fig
                
                # Print regression statistics for each outcome
                print("  Regression statistics:")
                for i, col in enumerate(cross_table.columns):
                    y_values = cross_table[col].values
                    regression = calculate_linear_regression(config['x_values'], y_values)
                    result_name = RESULT_CODE_MAPPING.get(col, f'Code {col}')
                    print(f"    {result_name:>10}: R² = {regression['r_squared']:.3f}, Slope = {regression['slope']:+.4f}")
                
                print("  ✓ Plot created successfully")
                
            except Exception as e:
                print(f"  ✗ Error creating plot for {var_name}: {e}")
                continue
                
        else:
            print(f"  ⚠ No configuration found for {var_name}")
        
        print("-" * 50)
    
    print(f"\n✓ Created {len(figures)} regression plots")
    return figures


def create_summary_statistics_table(cross_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table of regression statistics for all variables.
    
    Args:
        cross_tables: Dictionary of cross-tabulation tables
        
    Returns:
        pd.DataFrame: Summary statistics table
    """
    summary_data = []
    
    for var_name, cross_table in cross_tables.items():
        if var_name in VARIABLE_CONFIGS:
            config = VARIABLE_CONFIGS[var_name]
            x_values = config['x_values']
            
            if var_name == 'region':
                x_values = list(range(len(cross_table)))
            
            for col in cross_table.columns:
                y_values = cross_table[col].values
                regression = calculate_linear_regression(x_values, y_values)
                
                summary_data.append({
                    'Variable': var_name.replace('_', ' ').title(),
                    'Outcome': RESULT_CODE_MAPPING.get(col, f'Code {col}'),
                    'R_Squared': regression['r_squared'],
                    'Slope': regression['slope'],
                    'Interpretation': 'Positive' if regression['slope'] > 0 else 'Negative'
                })
    
    return pd.DataFrame(summary_data)


def create_student_segments_plot(segments: Dict) -> go.Figure:
    """
    Create a visualization of student segments.
    
    Args:
        segments: Student segments data
        
    Returns:
        go.Figure: Plotly figure showing student segments
    """
    # Extract data for plotting
    segment_names = []
    percentages = []
    completion_rates = []
    
    for segment_name, stats in segments.items():
        segment_names.append(segment_name.replace('_', ' ').title())
        percentages.append(stats['percentage'])
        completion_rates.append(stats.get('completion_rate', 0))
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Student Segment Sizes (%)', 'Completion Rates by Segment (%)'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add percentage bar chart
    fig.add_trace(
        go.Bar(
            x=segment_names,
            y=percentages,
            name='Percentage of Students',
            marker_color='lightblue',
            text=[f'{p:.1f}%' for p in percentages],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add completion rates bar chart
    fig.add_trace(
        go.Bar(
            x=segment_names,
            y=completion_rates,
            name='Completion Rate',
            marker_color='lightcoral',
            text=[f'{c:.1f}%' for c in completion_rates],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Student Segments", row=1, col=1)
    fig.update_xaxes(title_text="Student Segments", row=1, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Completion Rate (%)", row=1, col=2)
    
    fig.update_layout(
        title_text="Student Segment Analysis",
        height=500,
        showlegend=False
    )
    
    return fig


def create_correlation_heatmap(summary_stats: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap of R-squared values.
    
    Args:
        summary_stats: Summary statistics DataFrame
        
    Returns:
        go.Figure: Heatmap figure
    """
    if summary_stats.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="No data available for correlation heatmap")
        return fig
    
    try:
        # Pivot the data to create a matrix
        heatmap_data = summary_stats.pivot(index='Variable', columns='Outcome', values='R_Squared')
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu_r',
            text=heatmap_data.values.round(3),
            texttemplate='%{text}',
            colorbar=dict(title="R-squared")
        ))
        
        fig.update_layout(
            title='Correlation Strength (R²) Between Variables and Outcomes',
            xaxis_title='Student Outcomes',
            yaxis_title='Categorical Variables',
            height=500
        )
        
    except Exception as e:
        print(f"Warning: Could not create heatmap: {e}")
        fig = go.Figure()
        fig.update_layout(title="Error creating correlation heatmap")
    
    return fig


def save_all_plots(figures: Dict[str, go.Figure], output_dir: str = '../Notebooks/plots') -> None:
    """
    Save all plots as HTML files.
    
    Args:
        figures: Dictionary of Plotly figures
        output_dir: Directory to save plots (relative to Python_files)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving plots to {output_dir}...")
        
        saved_count = 0
        for var_name, fig in figures.items():
            try:
                filename = f"{var_name}_regression_plot.html"
                filepath = os.path.join(output_dir, filename)
                fig.write_html(filepath)
                print(f"  ✓ Saved {filename}")
                saved_count += 1
            except Exception as e:
                print(f"  ✗ Error saving {var_name}: {e}")
        
        print(f"✓ Successfully saved {saved_count} out of {len(figures)} plots")
        
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")


def create_all_visualizations(cross_tables: Dict[str, pd.DataFrame], 
                            summary_stats: pd.DataFrame,
                            segments: Dict) -> Dict[str, go.Figure]:
    """
    Create all visualizations for the analysis.
    
    Args:
        cross_tables: Cross-tabulation tables
        summary_stats: Summary statistics
        segments: Student segments data
        
    Returns:
        Dict[str, go.Figure]: All created figures
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Setup matplotlib style
    setup_matplotlib_style()
    
    # Create main regression plots
    figures = plot_all_cross_tables(cross_tables)
    
    # Create additional plots
    print("\nCreating additional visualizations...")
    
    try:
        # Student segments plot
        print("  Creating student segments plot...")
        segments_fig = create_student_segments_plot(segments)
        figures['student_segments'] = segments_fig
        print("  ✓ Student segments plot created")
    except Exception as e:
        print(f"  ✗ Error creating segments plot: {e}")
    
    try:
        # Correlation heatmap
        print("  Creating correlation heatmap...")
        heatmap_fig = create_correlation_heatmap(summary_stats)
        figures['correlation_heatmap'] = heatmap_fig
        print("  ✓ Correlation heatmap created")
    except Exception as e:
        print(f"  ✗ Error creating heatmap: {e}")
    
    print(f"\n✓ Created {len(figures)} visualizations total")
    
    return figures


def display_figures_in_notebook(figures: Dict[str, go.Figure]) -> None:
    """
    Display all figures in a notebook environment.
    
    Args:
        figures: Dictionary of Plotly figures
    """
    print("Displaying all figures...")
    
    for name, fig in figures.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print("-" * 40)
        fig.show()


if __name__ == "__main__":
    print("Visualization module ready for import")
    print("Use create_all_visualizations() to create all plots")
    print("Use plot_all_cross_tables() for regression plots only")