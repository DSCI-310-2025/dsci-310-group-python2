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

    # Load the dataset
    bill_data = pd.read_csv(input_path)

    # the following data validation checks if there are:
    # correct column names
    # correct data types in each column
    # no empty/null observations
    # missingness beyond certain threshold (given threshold is currently set at 0%)
    # no outlier or anomalous values (by checking if each variable is in the given range)
    # Target/response variable follows expected distribution (by checking class imbalance is within expected range)
    # No anomalous correlations between target and explanatory variables
    # No anomalous correlations between explanatory variable pairs
    schema = pa.DataFrameSchema({
        "variance": pa.Column(float, checks=[Check.gt(-10), Check.lt(10)], nullable=False), 
        "skewness": pa.Column(float, checks=[Check.gt(-15), Check.lt(15)], nullable=False),
        "curtosis": pa.Column(float, checks=[Check.gt(-20), Check.lt(20)], nullable=False),
        "entropy": pa.Column(float, checks=[Check.gt(-10), Check.lt(10)], nullable=False),
        "class": pa.Column(int, checks=[Check.isin([0, 1])], nullable=False)
    }, checks=[
        Check(lambda df: check_class_imbalance(df, 0.4), element_wise=False),
        Check(lambda df: check_no_high_correlation(df, 0.8), element_wise=False),
        # check if the dataframe only has the 5 given columns
        Check(lambda df: df.shape[1] == 5, element_wise=False)
    ])

    # schema will throw an error if any of the validation checks fail
    schema.validate(bill_data)

    # Preprocess the dataset
    bill_data = check_missing_value(bill_data)
    # Save the cleaned data
    bill_data.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to {output_path}!")

def check_class_imbalance(df: pd.DataFrame, min_class_proportion: float = 0.4):
    class_counts = df["class"].value_counts(normalize=True)
    minority_class_proportion = class_counts.min()
    if minority_class_proportion < min_class_proportion:
        raise ValueError(f"Class imbalance detected: minority class proportion is {minority_class_proportion:.2f}, which is below the required {min_class_proportion:.2f}.")
    return True

def check_no_high_correlation(df: pd.DataFrame, threshold: float = 0.9):
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

if __name__ == "__main__":
    main()

# How to run this script from root directory
# python scripts/02-clean_data.py --input_path data/original/BankNote_Authentication.csv --output_path data/clean/BankNote_Authentication_Clean.csv
