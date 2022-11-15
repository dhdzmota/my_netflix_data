import logging
import os

from zipfile import ZipFile
from time import perf_counter


def extract_zip_files(file, extract_to):
    """
    Helper function to extract zip files from one zip folder into another
     folder.

    Parameters
    ----------
    file: str
        String of the path containing the zip file that will be extracted.
    extract_to:
        String of the path containing the new directory where all the
         extraction will be saved in.

    Returns
    -------
    None
    """
    tick = perf_counter()
    logging.info(f'Extracting files.')
    with ZipFile(file, 'r') as zip_f:
        zip_f.extractall(extract_to)
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(f'Extracting files took {time_it_took} seconds.')
    return None


def process():
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    raw_data_path = os.path.join(general_path, 'data/raw')
    file_name_path = os.path.join(raw_data_path, 'netflix-report.zip')
    new_folder_path = os.path.join(raw_data_path, 'netflix-report')
    extract_zip_files(file=file_name_path, extract_to=new_folder_path)


if __name__ == "__main__":
    process()
