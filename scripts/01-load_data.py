import click
import pandas as pd
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option("--repo_id", default=267, help="ID of the UCI repository dataset to load.")
@click.option("--output_path", required=True, help="Path to save the loaded dataset.")
def main(repo_id, output_path):

    """
Script: 01-load_data.py
Description:
This script loads a dataset from the UCI Machine Learning Repository using the `ucimlrepo` package and saves it as a CSV file to a specified location.

It is designed to be run from the command line using `click`.

Usage:
    python scripts/01-load_data.py --repo_id 267 --output_path data/original/BankNote_Authentication.csv

Arguments:
--repo_id: Optional. The ID of the UCI dataset to fetch. Default is 267.
--output_path: Required. The file path (including filename) to save the dataset as a CSV.

Output:
- A CSV file containing the original dataset.
- The first 5 rows printed to the console.
- A message confirming successful save.

Dependencies:
- click
- pandas
- ucimlrepo
    """


    # Load the dataset
    bill_data = fetch_ucirepo(id=repo_id).data["original"]

    # Save loaded data to a local file
    bill_data.to_csv(output_path, index=False)

    click.echo(f"First 5 records:\n{bill_data.head()}")
    click.echo("Data successfully saved!")

if __name__ == "__main__":
    main()

## How to run this script from root directory
# python scripts/01-load_data.py --repo_id 267 --output_path data/original/BankNote_Authentication.csv
