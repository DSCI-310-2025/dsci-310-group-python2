import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

    # Evaluate different k values using cross-validation
    neighbors = range(1, 26)
    cv_scores = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())

    # Plot cross-validation accuracy vs. number of neighbors
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, cv_scores, marker='o')
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Mean CV Accuracy")
    plt.title("KNN Accuracy vs. Number of Neighbors")
    plt.grid(True)
    
    # Create output directory if needed
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    plt.savefig(f"{output_prefix}_knn_cv.png")
    plt.close()

    # Choose the best k (using the maximum CV accuracy)
    best_k = neighbors[cv_scores.index(max(cv_scores))]
    click.echo(f"Best k found: {best_k}")

    # Train the final KNN classifier using best_k
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)

    # Evaluate on the test set
    test_accuracy = final_knn.score(X_test, y_test)
    click.echo(f"Test set accuracy: {test_accuracy}")

    # Generate predictions and evaluation metrics
    y_pred = final_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Plot and save the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    plt.close()

    # Save the classification report to a text file
    report_path = f"{output_prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    click.echo("Modeling artifacts saved!")
    click.echo(f"Confusion matrix saved to: {output_prefix}_confusion_matrix.png")
    click.echo(f"Classification report saved to: {report_path}")

if __name__ == "__main__":
    main()

# How to run this script from the root directory:
# python scripts/04-model.py --input_path data/clean/BankNote_Authentication_Clean.csv --output_prefix results/analysis/BankNote_Authentication_Analysis
