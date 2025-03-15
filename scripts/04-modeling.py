#!/usr/bin/env python
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

@click.command()
@click.option("--input_path", required=True, help="Path to the cleaned CSV dataset.")
@click.option("--output_prefix", required=True, help="Output file prefix for saving modeling results (e.g., results/modeling/BankNote_Model).")
def main(input_path, output_prefix):
    """
    Read cleaned banknote authentication data, perform KNN modeling with cross-validation,
    and output the results (CV plot, confusion matrix, classification report) to files.
    """
    # Load the cleaned data
    try:
        data = pd.read_csv(input_path)
        click.echo("Cleaned data loaded successfully.")
    except Exception as e:
        click.echo(f"Error loading cleaned data: {e}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")

    # Separate features and target variable
    X = data.drop('class', axis=1)
    y = data['class']

    # Split data into training (75%) and testing (25%) sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    click.echo("Data split into training and testing sets.")

    # Cross-validation: test K values from 1 to 25 using 5-fold CV
    neighbors_range = range(1, 26)
    cv_scores = []
    for k in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())

    # Plot the cross-validation accuracy vs. number of neighbors
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_range, cv_scores, marker='o')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean CV Accuracy')
    plt.title('KNN Accuracy vs. Number of Neighbors')
    plt.grid(True)
    cv_plot_file = f"{output_prefix}_cv_accuracy.png"
    plt.savefig(cv_plot_file)
    click.echo(f"Saved CV accuracy plot to: {cv_plot_file}")
    plt.close()

    # Select the best k (highest CV accuracy)
    best_k = neighbors_range[np.argmax(cv_scores)]
    click.echo(f"Selected best k: {best_k}")

    # Train the final KNN model with the best k
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    test_accuracy = final_knn.score(X_test, y_test)
    click.echo(f"Test set accuracy: {test_accuracy:.4f}")

    # Generate predictions and compute confusion matrix
    y_pred = final_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_plot_file = f"{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_plot_file)
    click.echo(f"Saved confusion matrix plot to: {cm_plot_file}")
    plt.close()

    # Generate and save the classification report (as both text and CSV)
    report_text = classification_report(y_test, y_pred)
    report_txt_file = f"{output_prefix}_classification_report.txt"
    with open(report_txt_file, "w") as f:
        f.write(report_text)
    click.echo(f"Saved classification report to: {report_txt_file}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_file = f"{output_prefix}_classification_report.csv"
    report_df.to_csv(report_csv_file)
    click.echo(f"Saved classification report CSV to: {report_csv_file}")

if __name__ == "__main__":
    main()
