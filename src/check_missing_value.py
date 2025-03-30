import click
import pandas as pd 

def check_missing_value(bill_data):
        
    """
    Checks if the data haas missing (NAN) values
    
    Parameters:
        “bill_data: a variable that stores a pandas DataFrame — which is a data table loaded from a CSV file.
    
    Output:
       If the dataset has NAN values, function prints a message in the terminal and filters the data to remove the NAN values

    """
        # Check if data has the right type
    if not isinstance(bill_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    # Check for missing values
    missing_values = bill_data.isna().sum().sum()

    if missing_values:
        click.echo(f"Found {missing_values} missing values in the data.")
        # Drop missing values
        bill_data.dropna(inplace=True) #drops any rows with Nan
        click.echo("Missing values dropped!")

        return bill_data


    return bill_data