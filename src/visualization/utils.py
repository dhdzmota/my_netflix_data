import pandas as pd
import unidecode


def create_calendar_pivot_table(netflix_data):
    """
    Obtains a pivot table with calendar-like features, where the columns are
     the months and the rows are the years. The value is the sum of the
     individual durations.
    Parameters
    ----------
    netflix_data: pd.DataFrame
        A data frame that must contain the start time and the duration.

    Returns
    -------
    calendarized: pd.DataFrame
        A data frame with columns as months and rows as year.
    """
    grouper = create_grouper(freq='M')
    calendar_year = netflix_data.groupby(grouper).duration.sum().reset_index()
    calendar_year.duration = calendar_year.duration
    calendar_year['month'] = calendar_year.start_time.apply(lambda x: x.month)
    calendar_year['year'] = calendar_year.start_time.apply(lambda x: x.year)
    calendarized = pd.pivot_table(
        calendar_year,
        index='year',
        columns=['month'],
        values='duration'
    )
    return calendarized


def get_pivoted_data(netflix_data):
    groupers = [
        create_grouper(), 'profile_name']
    _grouped_netflix_data = netflix_data.groupby(groupers)
    grouped_netflix_data = _grouped_netflix_data.duration.sum().reset_index()
    pivoted_data = pd.pivot_table(
        grouped_netflix_data,
        values='duration',
        columns='profile_name',
        index='start_time',
        fill_value=0,
    )
    pivoted_data['total'] = pivoted_data.sum(axis=1)

    for col in pivoted_data.columns:
        pivoted_data[f'_{col}'] = pivoted_data[col] / pivoted_data['total']
    return pivoted_data


def create_grouper(key='start_time', freq='2M'):
    """
    This function is practically an implementation of the pd.Grouper function.


    Parameters
    ----------
    key: str
        String that indicates which column use as key.
    freq: str
        String that indicates which frequency to use.

    Returns
    -------
    grouper: pd.Grouper
        Pandas grouper over time.
    """
    grouper = pd.Grouper(
        key=key,
        freq=freq,
        closed='left',
        label='left',
        convention='start'
    )
    return grouper


def clean_text(text):
    """
    This function lowers the string, strips the additional blank spaces, then
     replaces blank spaces with underscore, afterwards removes special
     characters and finally strips if any underscore was kept at the end.

    Parameters
    ----------
    text: str
        Desired string to clean.

    Returns
    -------
    text: str
        Cleaned string.

    """
    text = text.lower()
    text = text.strip()
    text = text.replace(' ', '_')
    text = ''.join(e for e in text if e.isalnum() or e == '_')
    text = text.strip('_')
    text = unidecode.unidecode(text)
    return text
