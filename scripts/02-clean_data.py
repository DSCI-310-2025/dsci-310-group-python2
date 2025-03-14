import click
import pandas as pd

@click.command()
@click.option("--input_path", required=True, help="Path to the input dataset.")
@click.option("--output_path", required=True, help="Path to save the cleaned dataset.")
def main(input_path, output_path):
    """Read, clean, and save the dataset."""

    # Load the dataset
    bill_data = pd.read_csv(input_path)

    # Check for missing values
    missing_values = bill_data.isna().sum().sum()
    if missing_values:
        click.echo(f"Found {missing_values} missing values in the data.")

        # Drop missing values
        bill_data.dropna(inplace=True)
        click.echo("Missing values dropped!")

    # Save the cleaned data
    bill_data.to_csv(output_path, index=False)
    click.echo(f"Cleaned data saved to {output_path}!")

if __name__ == "__main__":
    main()

# How to run this script from root directory
# python scripts/02-clean_data.py --input_path data/original/BankNote_Authentication.csv --output_path data/clean/BankNote_Authentication_Clean.csv
