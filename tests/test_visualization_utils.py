import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from matplotlib import image
import os
import tempfile
import sys
import pytest
from sklearn.datasets import make_classification

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions to test
from src.visualization_utils import plot_histogram

# Create a fixture for test data
@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df['class'] = y
    return df

# get directory for the sample test images
@pytest.fixture
def get_test_images_dir():
    test_dir = os.path.join(os.path.dirname(__file__), 'test_images/')
    return test_dir


class TestPlotHistogram:
    # test if histogram compiles
    def test_plot_histogram_basic(self, sample_data):
        feature = 'feature1'
        plot_histogram(sample_data, feature)

    # test if histogram returns an error if feature or classes are missing from dataset
    def test_plot_histogram_error(self, sample_data):

        # missing feature
        missing_feature = 'non_existent_feature'
        with pytest.raises(ValueError, match="Could not interpret value `non_existent_feature` for `x`. An entry with this name does not appear in `data`"):
            plot_histogram(sample_data, missing_feature)

        # missing class
        feature = 'feature1'
        missing_class = sample_data.drop(columns=['class'])
        with pytest.raises(ValueError, match="Could not interpret value `class` for `hue`. An entry with this name does not appear in `data`"):
            plot_histogram(missing_class, feature)

    # test if histogram saves to directory and matches test image
    def test_plot_histogram_with_saving(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        resulting_image = f"{get_test_images_dir}with_saving_figure_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_saving_figure')
            plot_histogram(sample_data, feature, output_prefix=output_prefix)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {output_file} was not created"
            diff = compare_images(resulting_image, generated_image, tol=1e-2)
            assert diff is None, f"Images are different: {diff}"

    # test if histogram with custom labels matches test image
    def test_plot_histogram_with_labels(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        labels = ['A (Tan Bars)', 'B (Blue Bars)']
        labels2 = ["These labels shouldn't match at all", 'As they are completely different']
        resulting_image = f"{get_test_images_dir}with_labels_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_labels')

            # test matching labels
            plot_histogram(sample_data, feature, labels=labels,  output_prefix=output_prefix)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {output_file} was not created"
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is None, f"Images are different: {diff}"

            # test differing labels
            plot_histogram(sample_data, feature, labels=labels2,  output_prefix=output_prefix)
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is not None, f"Test failed: Images should differ in labels but they don't. {diff}"

    # test if histogram with custom figsizes matches test image
    def test_plot_histogram_with_figsize(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        labels = ['A (Tan Bars)', 'B (Blue Bars)']
        figsize = (10, 10)
        resulting_image = f"{get_test_images_dir}with_figsize_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_labels')

            # test matching figsizes
            plot_histogram(sample_data, feature, labels=labels,  output_prefix=output_prefix, figsize=figsize)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {output_file} was not created"
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is None, f"Images are different: {diff}"
            
            # test differing figsizes
            plot_histogram(sample_data, feature, labels=labels,  output_prefix=output_prefix)
            img1 = image.imread(resulting_image)
            img2 = image.imread(generated_image)
            assert img1.shape != img2.shape, "Images have the same sizes"