import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import sys


# Add the project root to the Python path to be able to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our abstracted functions
from banknote_utils.modeling_utils import (
    evaluate_knn_cv,
    plot_knn_cv,
    train_knn_model,
    evaluate_model
)

from banknote_utils.ensure_output_directory import ensure_output_directory


@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned dataset.")
@click.option("--output_prefix", required=True, help="Prefix for the output modeling artifacts (figures and reports).")
def main(input_path, output_prefix):
    """
    Script: 04-modeling.py

    Description:
    This script performs classification modeling on the cleaned banknote dataset using a K-Nearest Neighbors (KNN) algorithm.
    It includes hyperparameter tuning via cross-validation, model training, evaluation, and artifact saving.

    Steps performed:
    1. Splits the dataset into training and test sets (stratified).
    2. Uses cross-validation to determine the best value of k.
    3. Trains a KNN model using the selected k.
    4. Evaluates the model on the test set.
    5. Saves evaluation results including:
       - Cross-validation accuracy plot
       - Confusion matrix
       - Classification report
       - Best k parameter used

    Usage:
        python scripts/04-modeling.py \
            --input_path data/clean/BankNote_Authentication_Clean.csv \
            --output_prefix results/analysis/BankNote_Authentication_Analysis

    Arguments:
    --input_path: Path to the cleaned dataset CSV.
    --output_prefix: Prefix used for saving all output artifacts.

    Output:
    - A cross-validation plot (PNG)
    - A confusion matrix plot (PNG)
    - A classification report (CSV)
    - A CSV file recording the selected best k value

    Dependencies:
    - click
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - banknote_utils.modeling_utils
    """
    # check correct file format for input path
    if not input_path.lower().endswith('.csv'):
        raise ValueError("Input file must have a '.csv' extension")
        
    # Load cleaned dataset
    df = pd.read_csv(input_path)
    
    # Split into features and target
    X = df.drop("class", axis=1)
    y = df["class"]
    
    # Split data into training and testing sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=1
    )
    
    # Create output directory if needed

    ensure_output_directory(output_prefix)
        
    plt.savefig(f"{output_prefix}_knn_cv.png")
    plt.close()
    
    # Step 1: Evaluate different k values using cross-validation
    neighbors, cv_scores, best_k = evaluate_knn_cv(X_train, y_train)
    
    # Step 2: Plot and save the CV accuracy vs. number of neighbors
    plot_knn_cv(
        neighbors, 
        cv_scores, 
        output_path=f"{output_prefix}_knn_cv.png"
    )
    
    click.echo(f"Best k found: {best_k}")
    
    # Step 3: Train the final KNN classifier using best_k
    final_knn = train_knn_model(X_train, y_train, best_k)
    
    # Step 4: Evaluate on the test set and create confusion matrix
    model_results = evaluate_model(
        final_knn, 
        X_test, 
        y_test, 
        output_path=f"{output_prefix}_confusion_matrix.png"
    )
    
    click.echo(f"Test set accuracy: {model_results['accuracy']}")
    
    # Save the classification report to a CSV file
    model_results['classification_report'].to_csv(f"{output_prefix}_classification_report.csv")
    
    # Save the parameters used for the KNN to a CSV file
    KNN_parameters = pd.DataFrame({'best_k': [best_k]})
    KNN_parameters.to_csv(f"{output_prefix}_KNN_parameters.csv", index=False)
    
    click.echo("Modeling artifacts saved!")
    click.echo(f"Confusion matrix saved to: {output_prefix}_confusion_matrix.png")
    click.echo(f"Classification report saved to: {output_prefix}_classification_report.csv")

if __name__ == "__main__":
    main()

# How to run this script from the root directory:
# python scripts/04-modeling.py --input_path data/clean/BankNote_Authentication_Clean.csv --output_prefix results/analysis/BankNote_Authentication_Analysis