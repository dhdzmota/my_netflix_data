import pandas as pd


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
    netflix_data = pd.read_csv(data_path)
    netflix_data.start_time = pd.to_datetime(netflix_data.start_time)
    netflix_data.end_time = pd.to_datetime(netflix_data.end_time)
    netflix_data.duration = netflix_data.duration/3600
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
    data = pd.read_csv(data_path)
    if sorted_by:
        data = data.sort_values(sorted_by, ascending=False)
    if limit_rows:
        data = data.iloc[:limit_rows]
    return data
