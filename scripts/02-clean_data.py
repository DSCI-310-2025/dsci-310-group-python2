import click
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from banknote_utils.check_missing_value import check_missing_value

@click.command()
@click.option("--input_path", required=True, help="Path to the input dataset.")
@click.option("--output_path", required=True, help="Path to save the cleaned dataset.")
def main(input_path, output_path):
    """
Script: 02-clean_data.py

Description:
This script reads a dataset from a specified CSV file, checks for missing values using a helper function,
and saves the cleaned dataset to a new location. It is intended to be run from the command line using Click.

Usage:
    python scripts/02-clean_data.py --input_path data/original/BankNote_Authentication.csv --output_path data/clean/BankNote_Authentication_Clean.csv

Arguments:
--input_path: Path to the input CSV file.
--output_path: Path to save the cleaned CSV file.

Output:
- A cleaned version of the dataset saved to the specified output path.
- A message confirming the output location is printed to the terminal.

Dependencies:
- click
- pandas
- banknote_utils.check_missing_value
    """

    # Load the dataset
    bill_data = pd.read_csv(input_path)

    # Check for missing values
    bill_data = check_missing_value(bill_data)
    # Save the cleaned data
    bill_data.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to {output_path}!")

if __name__ == "__main__":
    main()

# How to run this script from root directory
# python scripts/02-clean_data.py --input_path data/original/BankNote_Authentication.csv --output_path data/clean/BankNote_Authentication_Clean.csv
