import os
import pytest
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from check_missing_value import check_missing_value

@pytest.fixture
# create test clean dataset 
def sample_df_no_missing():
    return pd.DataFrame({
        "name": ["Alice", "Bob"],
        "amount": [25.5, 30.0]
    })

@pytest.fixture
# create test dataset with one missing value 
def sample_df_with_missing():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "amount": [25.5, None, 30.0]
    })

@pytest.fixture
#create test dataset with all missing values
def sample_df_with_all_rows_missing():
    return pd.DataFrame({
        "name": [None, None, None],
        "amount": [None, None, None]
    })

@pytest.fixture
#create test dataset with all missing values
def sample_not_df():
    return np.array([1,2,3])

#test if the dataframe is clean
def test_no_missing_value(sample_df_no_missing,capfd):
    result = check_missing_value(sample_df_no_missing.copy())
    assert result.equals(sample_df_no_missing)

    out, _ = capfd.readouterr()
    assert "missing values" not in out.lower()


#test if the dataframe has one row with missign value
def test_one_missing_value(sample_df_with_missing,capfd):
    result = check_missing_value(sample_df_with_missing.copy())
    out, _ = capfd.readouterr()

    assert result.shape[0] == 2 # one row should be dropped
    assert result["amount"].isna().sum() == 0  # No NaNs left
    assert "missing values" in out.lower()
    assert "Missing values dropped!" in out #Make sure it prints out the right message in the terminal

#test if the dataframe has all rows missing
def test_all_missing_value(sample_df_with_all_rows_missing,capfd):

    result = check_missing_value(sample_df_with_all_rows_missing.copy())
    out, _ = capfd.readouterr()

    assert result.empty
    assert "missing values" in out.lower()
    assert "dropped!" in out.lower()

#test if it has the right data type 
def test_wrong_type(sample_not_df):
       
       with pytest.raises(TypeError):
           check_missing_value(sample_not_df)


