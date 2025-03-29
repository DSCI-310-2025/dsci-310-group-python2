import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root to the Python path to be able to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from src.ensure_output_directory import (
    ensure_output_directory
)

from src.visualization_utils import (
    plot_histogram
)

@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned dataset.")
@click.option("--output_prefix", required=True, help="Prefix for the output EDA artifacts (figures and tables).")
def main(input_path, output_prefix):
    """
    Perform exploratory data analysis on the banknote dataset and save visualizations.
    
    This script creates a count table and histograms for the following features:
      - variance
      - skewness
      - curtosis
      - entropy
      
    Each figure is saved using the provided output prefix.
    """
    # Load data
    df = pd.read_csv(input_path)

    # Create output directory if it does not exist
    ensure_output_directory(output_prefix)

    # Create and save count table (class proportions)
    count_table = df.groupby('class').size().reset_index(name='Count')
    count_table['Percentage'] = 100 * count_table['Count'] / len(df)
    count_table.to_csv(f"{output_prefix}_count_table.csv", index=False)

    # List of features to visualize
    features = ['variance', 'skewness', 'curtosis', 'entropy']
    for feature in features:
        plot_histogram(df, feature, ['A (Tan Bars)', 'B (Blue Bars)'], output_prefix)

    click.echo(f"EDA artifacts saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main()

# How to run this script from the root directory:
# python scripts/03-visualization.py --input_path data/clean/BankNote_Authentication_Clean.csv --output_prefix results/eda/BankNote_Authentication_EDA

