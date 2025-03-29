import os

def ensure_output_directory(output_prefix):
    """
    Ensure that the directory for the given output prefix exists.
    
    Parameters:
        output_prefix (str): The prefix for output files, including the directory path.
    
    Raises:
        FileExistsError: If a file exists at the expected directory location.
    """
    out_dir = os.path.dirname(output_prefix)

    if out_dir:
        # Check if something exists at this path
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):  # If it's a file, raise an error
            raise FileExistsError(f"A file exists at the expected directory path: {out_dir}")

        os.makedirs(out_dir, exist_ok=True)  # Create directory if it doesn't exist
