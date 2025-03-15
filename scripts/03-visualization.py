#!/usr/bin/env python
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned dataset.")
@click.option("--output_prefix", required=True, help="Output file prefix for saving EDA outputs (e.g., results/eda/BankNote_EDA).")
def main(input_path, output_prefix):
    """
    Generate exploratory data visualizations and a summary table from the cleaned banknote data.
    """
    # Load the cleaned data
    try:
        data = pd.read_csv(input_path)
        click.echo("Cleaned data loaded successfully.")
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        return

    # Ensure output directory exists (based on output_prefix)
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")

    # List of features to plot
    features = ['variance', 'skewness', 'curtosis', 'entropy']
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x=feature, hue='class', element='bars', bins=30, kde=True)
        plt.xlabel(feature.capitalize())
        plt.ylabel("Count")
        plt.title(f"Distribution of {feature.capitalize()} Grouped by Authenticity")
        plt.legend(title="Class", labels=["Authentic (0)", "Fake (1)"])
        plot_file = f"{output_prefix}_{feature}.png"
        plt.savefig(plot_file)
        click.echo(f"Saved {feature} histogram to {plot_file}")
        plt.close()

    # Create a summary table (descriptive statistics) and save it
    summary = data.describe().T
    summary_file = f"{output_prefix}_summary.csv"
    summary.to_csv(summary_file)
    click.echo(f"Saved descriptive summary table to {summary_file}")

if __name__ == "__main__":
    main()
