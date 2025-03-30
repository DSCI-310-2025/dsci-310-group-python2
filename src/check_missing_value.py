import click
import pandas as pd 

def check_missing_value(bill_data):
    # Check for missing values
    missing_values = bill_data.isna().sum().sum()

    if missing_values:
        click.echo(f"Found {missing_values} missing values in the data.")

        # Drop missing values
        bill_data.dropna(inplace=True)
        click.echo("Missing values dropped!")
    return bill_data