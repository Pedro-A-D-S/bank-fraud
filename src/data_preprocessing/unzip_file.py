import os
import zipfile
import logging

def unzip_file(zip_file_path: str, extract_to: str, log_file: str) -> None:
    """
    Unzip a zip file to the specified directory.

    Parameters
    ----------
    zip_file_path : str
        Path to the zip file.
    extract_to : str
        Directory where the contents of the zip file will be extracted.
    log_file : str
        Path to the log file.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Unzipped {zip_file_path} to {extract_to}")
    except Exception as e:
        logger.exception(f"Failed to unzip {zip_file_path}: {e}")
    
if __name__ == "__main__":
    
    path_to_zip_file = '../../data/raw/archive.zip'
    extract_to_directory = '../../data/raw'
    log_file_path = 'logs/unzip_log.log'

    # Create the directory if it doesn't exist
    os.makedirs(extract_to_directory, exist_ok = True)
    os.makedirs('logs', exist_ok = True)

    unzip_file(path_to_zip_file, extract_to_directory, log_file_path)
    print("File unzipped successfully.")
