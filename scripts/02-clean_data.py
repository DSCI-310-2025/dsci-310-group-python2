import click
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from banknote_utils.check_missing_value import check_missing_value
import pandera as pa
from pandera import Check

@click.command()
@click.option("--input_path", required=True, help="Path to the input dataset.")
@click.option("--output_path", required=True, help="Path to save the cleaned dataset.")
def main(input_path, output_path):
    """Read, clean, and save the dataset."""

    # check correct file format for input path
    if not input_path.lower().endswith('.csv'):
        raise ValueError("Input file must have a '.csv' extension")
    
    # check correct file format for output path
    if not output_path.lower().endswith('.csv'):
        raise ValueError("Output file must have a '.csv' extension")
    
    # Load the dataset
    bill_data = pd.read_csv(input_path)

    # the following data validation checks if there are:
    # correct column names
    # correct data types in each column
    # no empty/null observations
    # missingness beyond certain threshold
    # no outlier or anomalous values (by checking if each variable is in the given range)
    # Target/response variable follows expected distribution (by checking class imbalance is within expected range)
    # No anomalous correlations between target and explanatory variables
    # No anomalous correlations between explanatory variable pairs
    schema = pa.DataFrameSchema({
        "variance": pa.Column(float, checks=[Check.gt(-10), Check.lt(10)]), 
        "skewness": pa.Column(float, checks=[Check.gt(-15), Check.lt(15)]),
        "curtosis": pa.Column(float, checks=[Check.gt(-20), Check.lt(20)]),
        "entropy": pa.Column(float, checks=[Check.gt(-10), Check.lt(10)]),
        "class": pa.Column(int, checks=[Check.isin([0, 1])])
    }, checks=[
        Check(lambda df: check_missing_data(df, 0.0), element_wise=False),
        Check(lambda df: check_class_imbalance(df, 0.4), element_wise=False),
        Check(lambda df: check_no_high_correlation(df, 0.8), element_wise=False),
        # check if the dataframe only has the 5 given columns
        Check(lambda df: df.shape[1] == 5, element_wise=False)
    ])

    # schema will throw an error if any of the validation checks fail
    schema.validate(bill_data)

    # Preprocess the dataset
    bill_data = check_missing_value(bill_data)
    
    # Validate the data again after preprocessing the data
    schema.validate(bill_data)

    # Save the cleaned data
    bill_data.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to {output_path}!")

def check_class_imbalance(df: pd.DataFrame, min_class_proportion: float = 0.4):
    """
    Check for class imbalances in the DataFrame and
    ensure no class has a proportion less than the specified threshold for class imbalances

    Parameters:
    - df: pandas DataFrame
    - threshold: float (default 0.4), the minimum proportion a class can have in the given dataframe

    Output:
    - ValueError: If any column exceeds the missing data threshold.
    - Returns True and passes otherwise
    """
    class_counts = df["class"].value_counts(normalize=True)
    minority_class_proportion = class_counts.min()
    if minority_class_proportion < min_class_proportion:
        raise ValueError(f"Class imbalance detected: minority class proportion is {minority_class_proportion:.2f}, which is below the required {min_class_proportion:.2f}.")
    return True

def check_no_high_correlation(df: pd.DataFrame, threshold: float = 0.9):
    """
    Check for correlations in the DataFrame and
    ensure no pair of exploratory/target variables have a correlation above
    the given threshold for correlation values

    Parameters:
    - df: pandas DataFrame
    - threshold: float (default 0.9), the maximum correlation each pair of variables can have

    Output:
    - ValueError: If any column exceeds the 
    - Returns True and passes otherwise
    """
    corr_matrix = df.corr()
    high_corr_pairs = []
    
    num_columns = corr_matrix.shape[0]
    for i in range(num_columns):
        for j in range(i + 1, num_columns): 
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((df.columns[i], df.columns[j]))

    if high_corr_pairs:
        raise ValueError(f"Anomalous correlation detected between the following variable pairs: {high_corr_pairs}.")
    return True

def check_missing_data(df, threshold=0.01):
    """
    Check for missing data in the DataFrame and ensure no column exceeds the specified threshold for missing values.

    Parameters:
    - df: pandas DataFrame
    - threshold: float (default 0.01), the maximum proportion of missing data allowed per column.

    Output:
    - ValueError: If any column exceeds the missing data threshold or an empty observation exists
    - Returns True and passes otherwise
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean()

    # Check if any column has missing values above the threshold
    columns_above_threshold = missing_percentage[missing_percentage > threshold]

    if not columns_above_threshold.empty:
        raise ValueError(f"Columns with missing data above threshold ({threshold*100}%):\n{columns_above_threshold}")
    
    # Check if there are any rows with missing values
    if df.isnull().any(axis=1).sum() > 0:
        raise ValueError("There are rows with missing (empty) observations.")
    
    return True

if __name__ == "__main__":
    main()

# How to run this script from root directory
# python scripts/02-clean_data.py --input_path data/original/BankNote_Authentication.csv --output_path data/clean/BankNote_Authentication_Clean.csv
