import os
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ensure_output_directory import ensure_output_directory  

@pytest.fixture
def temp_output_prefix(tmp_path):
    """Provides a temporary directory for testing."""
    return os.path.join(tmp_path, "test_output")

def test_creates_new_directory(temp_output_prefix):
    """Test that the function creates a directory when it doesn't exist."""
    ensure_output_directory(temp_output_prefix)
    assert os.path.exists(os.path.dirname(temp_output_prefix))

def test_existing_directory(temp_output_prefix):
    """Test that the function does nothing if the directory already exists."""
    dir_path = os.path.dirname(temp_output_prefix)

    # Ensure directory exists before calling function
    os.makedirs(dir_path, exist_ok=True)  
    assert os.path.isdir(dir_path)  # Ensure it's actually a directory

    ensure_output_directory(temp_output_prefix)
    assert os.path.exists(dir_path)  # Directory should still exist

def test_empty_prefix():
    """Test that the function does nothing when given an empty string."""
    ensure_output_directory("")  # Should not raise an error

def test_conflicting_file(tmp_path):
    """Test that the function raises an error if a file exists at the expected directory path."""
    conflicting_dir_path = tmp_path / "conflicting_dir"

    # Create a file where the directory should be
    conflicting_dir_path.write_text("This is a file, not a directory.")

    # Use a fake output prefix that expects conflicting_dir_path to be a directory
    output_prefix = str(conflicting_dir_path / "output_file.csv")

    with pytest.raises(FileExistsError, match="A file exists at the expected directory path"):
        ensure_output_directory(output_prefix)  # Should raise an error

def test_root_directory(tmp_path):
    """Test that the function does nothing if the output is in a root directory."""
    output_prefix = str(tmp_path / "output_file.csv")  # Use tmp_path as root dir

    ensure_output_directory(output_prefix)  # Should NOT raise an error

    assert tmp_path.exists() and tmp_path.is_dir()  # Root dir should still exist