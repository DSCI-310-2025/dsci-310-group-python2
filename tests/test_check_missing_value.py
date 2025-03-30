import os
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from check_missing_value import check_missing_value

@pytest.fixture
# create test clean dataset 
def sample_df_no_missing():
    return pd.DataFrame({
        "name": ["Alice", "Bob"],
        "amount": [25.5, 30.0]
    })

# create test dataset with one missing value 
def sample_df_with_missing():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "amount": [25.5, None, 30.0]
    })

#create test dataset with all missing values
def sample_df_with_missing():
    return pd.DataFrame({
        "name": [None, None, None],
        "amount": [None, None, None]
    })

def test_no_missing_value:
    