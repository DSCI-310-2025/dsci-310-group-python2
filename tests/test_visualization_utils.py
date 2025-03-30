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
from src.visualization_utils import (plot_histogram, create_count_table)

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

# tests for histogram plotting function
class TestPlotHistogram:
    # test if histogram compiles
    def test_plot_histogram_basic(self, sample_data):
        feature = 'feature1'
        plot_histogram(sample_data, feature, "class")
        
        # use unknown_variable as target variable instead of 'class'
        sample_data.rename(columns={'class': 'unknown_variable'}, inplace=True)
        plot_histogram(sample_data, feature, "unknown_variable")

    # test if histogram returns an error if feature or classes are missing from dataset
    def test_plot_histogram_error(self, sample_data):

        # missing feature
        missing_feature = 'non_existent_feature'
        with pytest.raises(ValueError, match="Could not interpret value `non_existent_feature` for `x`. An entry with this name does not appear in `data`"):
            plot_histogram(sample_data, missing_feature, "class")

        # missing target_variable
        feature = 'feature1'
        with pytest.raises(ValueError, match="Could not interpret value `missing_class` for `hue`. An entry with this name does not appear in `data`"):
            plot_histogram(sample_data, feature, "missing_class")

        # empty dataframe case
        empty_df = pd.DataFrame(columns=["feature1", 'lacking_class'])
        with pytest.raises(ValueError, match="`dataset` input should have multiple elements."):
            plot_histogram(empty_df, feature, "lacking_class")

        # dataframe with only one item
        single_item_df = pd.DataFrame({"feature1": [1.0], "lacking_class": ["A"]})
        with pytest.raises(ValueError, match="`dataset` input should have multiple elements."):
            plot_histogram(single_item_df, feature, "lacking_class")

    # test if histogram saves to directory and matches test image
    def test_plot_histogram_with_saving(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        resulting_image = f"{get_test_images_dir}with_saving_figure_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_saving_figure')
            plot_histogram(sample_data, feature, "class", output_prefix=output_prefix)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {generated_image} was not created"
            diff = compare_images(resulting_image, generated_image, tol=1e-2)
            assert diff is None, f"Test failed: Images are different when should have the same default parameters: {diff}"

    # test if histogram with custom labels matches test image
    def test_plot_histogram_with_labels(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        labels = ['A (Tan Bars)', 'B (Blue Bars)']
        labels2 = ["These labels shouldn't match at all", 'As they are completely different']
        resulting_image = f"{get_test_images_dir}with_labels_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_labels')

            # test matching labels
            plot_histogram(sample_data, feature, "class", labels=labels,  output_prefix=output_prefix)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {generated_image} was not created"
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is None, f"Test failed: Images are different when they should have the same label: {diff}"

            # test differing labels
            plot_histogram(sample_data, feature, "class", labels=labels2,  output_prefix=output_prefix)
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is not None, f"Test failed: Images are the same when they should differ in labels"

    # test if histogram with custom figsizes matches test image
    def test_plot_histogram_with_figsize(self, sample_data, get_test_images_dir):
        feature = 'feature1'
        labels = ['A (Tan Bars)', 'B (Blue Bars)']
        figsize = (10, 10)
        resulting_image = f"{get_test_images_dir}with_figsize_{feature}.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_prefix = os.path.join(temp_dir, 'with_labels')

            # test matching figsizes
            plot_histogram(sample_data, feature, "class", labels=labels,  output_prefix=output_prefix, figsize=figsize)
            generated_image = f"{output_prefix}_{feature}.png"
            assert os.path.exists(generated_image), f"Output file {generated_image} was not created"
            diff = compare_images(resulting_image, generated_image, tol=0)
            assert diff is None, f"Test failed: Images are different when they should have the same figure size: {diff}"
            
            # test differing figsizes
            plot_histogram(sample_data, feature, "class", labels=labels,  output_prefix=output_prefix)
            img1 = image.imread(resulting_image)
            img2 = image.imread(generated_image)
            assert img1.shape != img2.shape, "Test failed: Images have the same sizes when they should have different figure sizes"

# tests for create_count_table function
class TestCreateCountTable:
    # test if count table is created & compiles
    def test_basic_count_table(self, sample_data):
        create_count_table(sample_data, "class")

    # test if target class exists and returns error if not
    def test_count_table_error(self, sample_data):
        wrong_class_name = "wrong_target"
        
        with pytest.raises(KeyError, match=wrong_class_name):
            create_count_table(sample_data, wrong_class_name)

    # test whether count table saves correctly
    def test_count_table_with_saving(self, sample_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            # check if table has 2 classes & 3 coulmns denoting count table information
            create_count_table(sample_data, "class", temp_dir)
            generated_table = f"{temp_dir}_count_table.csv"
            assert os.path.exists(generated_table), f"Output file {generated_table} was not created"
            table = pd.read_csv(generated_table)
            assert table.shape == (2, 3), f"Shape mismatch: Expected (2, 3), but got {table.shape}"

            # create a new row & check if the newly created table has 3 classes
            new_row = {'feature1': 1, 'feature2': 2, 'feature3': 3, 'feature4': 4, 'class': 30}
            new_row_df = pd.DataFrame([new_row])
            additional_class_sample = pd.concat([sample_data, new_row_df], ignore_index=True)
            create_count_table(additional_class_sample, "class", temp_dir)
            additional_class_table =  pd.read_csv(generated_table)
            assert additional_class_table.shape == (3, 3), f"Shape mismatch: Expected (3, 3), but got {additional_class_table.shape}"

            # check case containing no rows
            empty_df = pd.DataFrame(columns=['lacking_class'])
            create_count_table(empty_df, "lacking_class", temp_dir)
            empty_class_table = pd.read_csv(generated_table)
            assert empty_class_table.shape == (0, 3), f"Shape mismatch: Expected (0, 3), but got {empty_class_table.shape}"

