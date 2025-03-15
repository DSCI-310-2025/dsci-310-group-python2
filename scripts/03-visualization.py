import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create and save count table (class proportions)
    count_table = df.groupby('class').size().reset_index(name='Count')
    count_table['Percentage'] = 100 * count_table['Count'] / len(df)
    count_table.to_csv(f"{output_prefix}_count_table.csv", index=False)

    # List of features to visualize
    features = ['variance', 'skewness', 'curtosis', 'entropy']
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=feature, hue='class', element='bars', bins=30, kde=True)
        plt.xlabel(feature.capitalize())
        plt.ylabel("Count")
        plt.title(f"Distribution of {feature.capitalize()} Grouped by Authenticity")
        plt.legend(title="Class", labels=["1 (Fake)", "0 (Authentic)"])
        plt.tight_layout()
        # Save each figure with a descriptive filename
        plt.savefig(f"{output_prefix}_{feature}.png")
        plt.close()

    click.echo(f"EDA artifacts saved with prefix: {output_prefix}")

if __name__ == "__main__":
    main()

# How to run this script from the root directory:
# python scripts/03-visualization.py --input_path data/clean/BankNote_Authentication_Clean.csv --output_prefix results/eda/BankNote_Authentication_EDA

