import pandas as pd
import logging

from time import perf_counter


SECONDS_IN_HOUR = 3600


def get_processed_netflix_data(data_path):
    """
    Get the processed netflix data

    Parameters
    ----------
    data_path: str
        location of the interest netflix processed data.

    Returns
    -------
    netflix_data: pd.DataFrame
        A data frame with the processed netflix information.

    """
    logging.info('Reading netflix processed data.')
    netflix_data = pd.read_csv(data_path)
    logging.info('Making start_time into datetime.')
    netflix_data.start_time = pd.to_datetime(netflix_data.start_time)
    logging.info('Making end_time into datetime.')
    netflix_data.end_time = pd.to_datetime(netflix_data.end_time)
    logging.info('Dividing duration seconds over 3600 to get hours.')
    netflix_data.duration = netflix_data.duration/SECONDS_IN_HOUR
    return netflix_data


def get_general_sorted_data(data_path, sorted_by='', limit_rows=0):
    """
    Get a general DataFrame from a CSV (sorted and limited by rows)

    Parameters
    ----------
    data_path: str
        Location of the interest data.
    sorted_by: str
        String indicating if data must be sorted somehow in descending order.
    limit_rows: int
        Number of rows that should be shown.

    Returns
    -------
    data: pd.DataFrame
        DataFrame with the desired information.

    """
    logging.info(f'Reading processed data from {data_path}.')
    data = pd.read_csv(data_path)
    if sorted_by:
        logging.info(f'Data is sorted by {sorted_by}.')
        data = data.sort_values(sorted_by, ascending=False)
    if limit_rows:
        logging.info(f'Limit rows from data: {limit_rows} rows.')
        data = data.iloc[:limit_rows]
    return data
