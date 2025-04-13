import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy import stats

# Add the project root to the Python path to be able to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from banknote_utils.ensure_output_directory import ensure_output_directory


from banknote_utils.visualization_utils import (
    plot_histogram,
    create_count_table
)

@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned dataset.")
@click.option("--output_prefix", required=True, help="Prefix for the output EDA artifacts (figures and tables).")
def main(input_path, output_prefix):
    """
    Script: 03-visualization.py

    Description:
    This script performs exploratory data analysis (EDA) on the cleaned banknote dataset. It generates and saves:
    - A count table showing class distributions
    - Histograms for four numeric features: variance, skewness, curtosis, and entropy

    Usage:
        python scripts/03-visualization.py \
            --input_path data/clean/BankNote_Authentication_Clean.csv \
            --output_prefix results/eda/BankNote_Authentication_EDA

    Arguments:
    --input_path: Path to the cleaned CSV dataset.
    --output_prefix: Prefix used to name and save all generated visualizations and tables.

    Output:
    - A CSV count table saved to the specified output path.
    - A CSV table showing the p-values for hypothesis testing for each explanatory variable compared to the target variable
    - Four histogram image files (one per feature), saved using the given prefix.
    - A terminal message confirming where the artifacts were saved.

    Dependencies:
    - click
    - pandas
    - seaborn
    - matplotlib
    - banknote_utils.visualization_utils
    """
    # check correct file format for input path
    if not input_path.lower().endswith('.csv'):
        raise ValueError("Input file must have a '.csv' extension")
    
    # Load data
    df = pd.read_csv(input_path)

    # Create and save count table (class proportions)
    create_count_table(df, "class", output_prefix)

    # List of features to visualize
    features = ['variance', 'skewness', 'curtosis', 'entropy']
    for feature in features:
        plot_histogram(df, feature, "class", ['A (Tan Bars)', 'B (Blue Bars)'], output_prefix)

    click.echo(f"EDA artifacts saved with prefix: {output_prefix}")
    
    # perform hypothesis test
    p_values = []
    
    for feature in df.columns[:-1]:
        group_0 = df[df['class'] == 0][feature]
        group_1 = df[df['class'] == 1][feature]
        
        t_stat, p_value = stats.ttest_ind(group_0, group_1)
        
        p_values.append({'Feature': feature, 'p-value': p_value})
        
    p_values_df = pd.DataFrame(p_values)
    p_values_df['p-value'] = p_values_df['p-value'].apply(lambda x: f'{x:.10f}')
    p_values_df.to_csv(f"{output_prefix}_p_values.csv", index=False)
    
    click.echo(f"P-value Results saved to: {output_prefix}_p_values.csv")

if __name__ == "__main__":
    main()
