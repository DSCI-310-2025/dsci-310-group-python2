import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import ensure_output_directory

def plot_histogram(dataset, feature, labels=None, output_prefix=None, figsize=(8, 6)):
    """
    Creates a basic histogram for the given dataset using the given feature
    
    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        DataFrame containing the given feature to plot a histogram for
    feature : pandas.Series or numpy.ndarray
        Target variable to plot a histogram for
    labels : list of strings, optional
        Add custom labels for the histogram's legend. If None, use default labels.
    output_path : str, optional
        Path to save the histogram. If None, the plot is not saved.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (8, 6).
    
    Returns
    -------
    None

    If a output_prefix is given, the function saves a plot to the given file location
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=dataset, x=feature, hue='class', element='bars', bins=30, kde=True)
    plt.xlabel(feature.capitalize())
    plt.ylabel("Count")
    if labels:
        plt.legend(title="Class", labels=labels)
    plt.tight_layout()
    if output_prefix:
        ensure_output_directory.ensure_output_directory(output_prefix)
        plt.savefig(f"{output_prefix}_{feature}.png")
    plt.close()