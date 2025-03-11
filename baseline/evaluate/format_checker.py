import os
import argparse
import logging
import pandas as pd

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
COLUMNS = ['id', 'label']

def check_format(file_path: str) -> bool:
    """
    Check whether the CSV file at file_path is in the correct format.
    The file must contain the columns 'id' and 'label'. It also checks for missing values
    and verifies that for subtask A the labels are either 0 or 1.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return False

    try:
        df = pd.read_csv(file_path)[COLUMNS]
    except Exception as e:
        logging.error(f"Error reading CSV file {file_path}: {e}")
        return False

    for column in COLUMNS:
        if df[column].isna().any():
            logging.error(f"NA value found in file {file_path} in column {column}")
            return False

    # For subtask A, valid labels are 0 or 1.
    if not df['label'].isin(["human", "machine"]).all():
        logging.error(f"Unknown label in file {file_path}. Unique labels: {df['label'].unique()}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file_path", "-p", nargs='+', required=True,
                        help="Path(s) to the CSV file(s) you want to check.", type=str)
    args = parser.parse_args()
    
    logging.info(f"Checking files: {args.prediction_file_path}")
    
    for file_path in args.prediction_file_path:
        valid = check_format(file_path)
        result = 'Format is correct' if valid else 'Something wrong in file format'
        logging.info(f"Checking file: {file_path}. Result: {result}")
