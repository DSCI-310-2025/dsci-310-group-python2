import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from src import ensure_output_directory

def evaluate_knn_cv(X_train, y_train, k_range=range(1, 26)):
    """
    Evaluate different k values for KNN using cross-validation.
    
    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features.
    y_train : pandas.Series or numpy.ndarray
        Target variable for training.
    k_range : range or list, optional
        Range of k values to evaluate, by default range(1, 26).
    
    Returns
    -------
    tuple
        (neighbors, cv_scores, best_k) where:
        - neighbors is a list of k values evaluated
        - cv_scores is a list of corresponding mean CV scores
        - best_k is the k value with the highest CV score
    """
    if not isinstance(k_range, (range, list)):
        raise TypeError("k_range must be a range or list")
    if len(k_range) == 0:
        raise ValueError("k_range cannot be empty")
        
    neighbors = list(k_range)
    cv_scores = []
    
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    
    # Find the best k (maximum CV accuracy)
    best_k = neighbors[cv_scores.index(max(cv_scores))]
    
    return neighbors, cv_scores, best_k


def plot_knn_cv(neighbors, cv_scores, output_path=None, figsize=(10, 6)):
    """
    Plot the cross-validation accuracy vs. number of neighbors.
    
    Parameters
    ----------
    neighbors : list or array-like
        List of k values evaluated.
    cv_scores : list or array-like
        List of corresponding mean CV scores.
    output_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (10, 6).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    if len(neighbors) != len(cv_scores):
        raise ValueError("neighbors and cv_scores must have the same length")
    
    plt.figure(figsize=figsize)
    fig = plt.gcf()
    plt.plot(neighbors, cv_scores, marker='o')
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Mean CV Accuracy")
    plt.grid(True)
    
    if output_path:
        ensure_output_directory.ensure_output_directory(output_path)
            
        plt.savefig(output_path)
    
    return fig


def train_knn_model(X_train, y_train, k):
    """
    Train a KNN classifier with the specified k.
    
    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features.
    y_train : pandas.Series or numpy.ndarray
        Target variable for training.
    k : int
        Number of neighbors for KNN.
    
    Returns
    -------
    sklearn.neighbors.KNeighborsClassifier
        Trained KNN model.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
        
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    return knn


def evaluate_model(model, X_test, y_test, output_path=None, figsize=(6, 5)):
    """
    Evaluate a trained model and create a confusion matrix plot.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict method.
    X_test : pandas.DataFrame or numpy.ndarray
        Test features.
    y_test : pandas.Series or numpy.ndarray
        True target values.
    output_path : str, optional
        Path to save the confusion matrix plot. If None, the plot is not saved.
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (6, 5).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'accuracy': Test set accuracy
        - 'confusion_matrix': Confusion matrix
        - 'classification_report': DataFrame with the classification report
        - 'y_pred': Predicted values
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    fig = plt.gcf()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    if output_path:
        ensure_output_directory.ensure_output_directory(output_path)
            
        plt.savefig(output_path)
    
    return {
        'accuracy': test_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'figure': fig
    }