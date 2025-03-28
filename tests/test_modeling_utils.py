import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tempfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from src.modeling_utils import (
    evaluate_knn_cv,
    plot_knn_cv,
    train_knn_model,
    evaluate_model
)

# Create a fixture for test data
@pytest.fixture
def test_data():
    # Generate a simple classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    
    # Convert to pandas DataFrame and Series (like our actual data)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='class')
    
    # Split data (70% train, 30% test)
    train_indices = np.random.choice(len(X), size=int(0.7 * len(X)), replace=False)
    test_indices = np.array([i for i in range(len(X)) if i not in train_indices])
    
    X_train = X_df.iloc[train_indices]
    y_train = y_series.iloc[train_indices]
    X_test = X_df.iloc[test_indices]
    y_test = y_series.iloc[test_indices]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


class TestEvaluateKnnCV:
    def test_basic_functionality(self, test_data):
        """Test that evaluate_knn_cv returns the expected structure."""
        neighbors, cv_scores, best_k = evaluate_knn_cv(
            test_data['X_train'], 
            test_data['y_train'],
            k_range=range(1, 5)
        )
        
        # Check types and shapes
        assert isinstance(neighbors, list)
        assert isinstance(cv_scores, list)
        assert isinstance(best_k, int)
        assert len(neighbors) == len(cv_scores) == 4
        assert best_k in neighbors
        
        # Check values are within expected range
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_invalid_k_range(self, test_data):
        """Test that evaluate_knn_cv raises appropriate errors for invalid input."""
        # Test with empty range
        with pytest.raises(ValueError, match="k_range cannot be empty"):
            evaluate_knn_cv(test_data['X_train'], test_data['y_train'], k_range=[])
        
        # Test with invalid type
        with pytest.raises(TypeError, match="k_range must be a range or list"):
            evaluate_knn_cv(test_data['X_train'], test_data['y_train'], k_range=42)


class TestPlotKnnCV:
    def test_basic_functionality(self):
        """Test that plot_knn_cv creates and returns a figure."""
        neighbors = [1, 2, 3, 4, 5]
        cv_scores = [0.8, 0.85, 0.9, 0.85, 0.8]
        
        fig = plot_knn_cv(neighbors, cv_scores)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    def test_save_to_file(self):
        """Test that plot_knn_cv saves to file when output_path is provided."""
        neighbors = [1, 2, 3, 4, 5]
        cv_scores = [0.8, 0.85, 0.9, 0.85, 0.8]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'test_plot.png')
            fig = plot_knn_cv(neighbors, cv_scores, output_path=output_path)
            
            # Check that file was created
            assert os.path.exists(output_path)
            plt.close(fig)  # Clean up
    
    def test_mismatched_lengths(self):
        """Test that plot_knn_cv raises error when inputs have different lengths."""
        neighbors = [1, 2, 3, 4, 5]
        cv_scores = [0.8, 0.85, 0.9]
        
        with pytest.raises(ValueError, match="neighbors and cv_scores must have the same length"):
            plot_knn_cv(neighbors, cv_scores)


class TestTrainKnnModel:
    def test_basic_functionality(self, test_data):
        """Test that train_knn_model returns a trained KNN model."""
        model = train_knn_model(test_data['X_train'], test_data['y_train'], k=3)
        
        assert isinstance(model, KNeighborsClassifier)
        assert model.n_neighbors == 3
        assert hasattr(model, 'predict')
        
        # Test that the model can make predictions
        predictions = model.predict(test_data['X_test'])
        assert len(predictions) == len(test_data['X_test'])
    
    def test_invalid_k(self, test_data):
        """Test that train_knn_model raises appropriate errors for invalid k."""
        # Test with non-integer k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            train_knn_model(test_data['X_train'], test_data['y_train'], k=3.5)
        
        # Test with non-positive k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            train_knn_model(test_data['X_train'], test_data['y_train'], k=0)


class TestEvaluateModel:
    def test_basic_functionality(self, test_data):
        """Test that evaluate_model returns the expected structure."""
        # Train a simple model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(test_data['X_train'], test_data['y_train'])
        
        # Evaluate the model
        results = evaluate_model(model, test_data['X_test'], test_data['y_test'])
        
        # Check that the result has all expected keys
        expected_keys = ['accuracy', 'confusion_matrix', 'classification_report', 'y_pred', 'figure']
        for key in expected_keys:
            assert key in results
        
        # Check types
        assert isinstance(results['accuracy'], float)
        assert isinstance(results['confusion_matrix'], np.ndarray)
        assert isinstance(results['classification_report'], pd.DataFrame)
        assert isinstance(results['y_pred'], np.ndarray)
        assert isinstance(results['figure'], plt.Figure)
        
        # Check that the accuracy is within the expected range
        assert 0 <= results['accuracy'] <= 1
        
        # Check that the confusion matrix has the right shape
        assert results['confusion_matrix'].shape == (2, 2)  # Binary classification
        
        # Close the figure to clean up
        plt.close(results['figure'])
    
    def test_save_to_file(self, test_data):
        """Test that evaluate_model saves to file when output_path is provided."""
        # Train a simple model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(test_data['X_train'], test_data['y_train'])
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, 'test_cm.png')
            results = evaluate_model(
                model, 
                test_data['X_test'], 
                test_data['y_test'],
                output_path=output_path
            )
            
            # Check that file was created
            assert os.path.exists(output_path)
            plt.close(results['figure'])  # Clean up