import os

def ensure_output_directory(output_prefix):
    """
    Ensure that the directory for the given output prefix exists.
    
    Parameters:
        output_prefix (str): The prefix for output files, including the directory path.
    """
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
