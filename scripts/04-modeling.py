import click
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys

# Add the project root to the Python path to be able to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our abstracted functions
from src.modeling_utils import (
    evaluate_knn_cv,
    plot_knn_cv,
    train_knn_model,
    evaluate_model
)

@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned dataset.")
@click.option("--output_prefix", required=True, help="Prefix for the output modeling artifacts (figures and reports).")
def main(input_path, output_prefix):
    """
    Perform modeling on the banknote dataset using KNN and save results.
    
    The script:
      - Splits the data into training and test sets.
      - Uses cross-validation to select the best value for k.
      - Trains a KNN classifier and evaluates its performance.
      - Saves a CV accuracy plot, confusion matrix plot, and a classification report.
    """
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
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
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