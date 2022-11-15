import datetime
import logging
import os
import pandas as pd

from time import perf_counter


def process_netflix_data(df):
    """
    This function makes a transformation of the raw data given by netflix by
    applying certain relevant steps:
        1) Transformation of column names (makes easier the manipulation for
            further dataframe operations).
        2) Anonymize the profiles (relevant if personal information is a deal).
        3) Transform object of dates into real datetime objects.
        4) Transform object of duration into a float value.
        5) Generate additional columns: end_time, new_title
        6) Drop non-used columns.

    Parameters
    ----------
    df: pd.DataFrame
        The raw data given by netflix.

    Returns
    -------
    netflix_data: pd.DataFrame
        Processed netflix data.

    """
    # Transform columns for an easier manipulation
    tick = perf_counter()
    logging.info(f'Renaming columns.')

    new_columns = {
        col: col.lower().strip().replace(' ', '_')
        for col in df.columns
    }
    logging.info(f'Renaming columns: {new_columns}.')
    df.rename(columns=new_columns, inplace=True)

    # Anonymize the different profiles
    profiles_dict = {
        profile_name: f'profile_{num}'
        for num, profile_name
        in enumerate(df.profile_name.unique())
    }
    logging.info(f'Renaming profile_name: {profiles_dict}.')
    df.profile_name = df.profile_name.apply(
        lambda x: profiles_dict[x]
    )
    # Transform star_time into a datetime
    df.start_time = pd.to_datetime(df.start_time)
    df.start_time = df.start_time  # - datetime.timedelta(hours=6)
    logging.info(f'Getting duration as time_delta.')

    df.duration = df.duration.apply(
        get_duration_timedelta
    )
    logging.info(f'Getting end_time from duration time_delta.')
    df['end_time'] = df.start_time + df.duration
    df.duration = df.duration.apply(
        lambda x: x.total_seconds()
    )
    logging.info(f'Changing title to new_title.')
    df['new_title'] = df.title.apply(lambda x: x.split(':')[0])

    logging.info(f'Removing non-played by profile.')
    df_no_auto_played = df[df.attributes.isna()]
    netflix_data = df_no_auto_played[
        df_no_auto_played.supplemental_video_type.isna()
    ]
    non_used_cols = ['attributes', 'supplemental_video_type']
    logging.info(f'Removing non_used_columns: {non_used_cols}.')

    netflix_data.drop(non_used_cols, axis=1, inplace=True)
    logging.info(f'Removing non_used_columns: {non_used_cols}.')
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(f'Processing netflix data took {time_it_took} seconds.')
    return netflix_data


def get_duration_timedelta(string_time=None, time_format="%H:%M:%S"):
    """
    This is a function to obtain a string that indicates duration as hh:mm:ss
     to an actual timedelta value.

    Parameters
    ----------
    string_time: str
        String that indicates a duration.
    time_format:
        Format in which the string is given. It will be by default "%H:%M:%S".

    Returns
    -------
    timedelta: datetime.timedelta
        converts the given duration to a time delta so it can be used in
        different datetime operations.

    """
    logging.info(
        f'Transforming the string time "{string_time}" into datetime.'
    )

    strptime = datetime.datetime.strptime(string_time, time_format).time()
    timedelta = datetime.timedelta(
        hours=strptime.hour,
        minutes=strptime.minute,
        seconds=strptime.second,
        microseconds=strptime.microsecond,
    )
    return timedelta


def get_netflix_data(data_path):
    """
    Get the netflix data and process it.

    Parameters
    ----------
    data_path: str
        location of the interest netflix data that will be processed.

    Returns
    -------
    netflix_data_with_series: pd.DataFrame
        A data frame with the information of the processed netflix data.

    """
    logging.info(f'Getting the netflix information')
    netflix_data_all = pd.read_csv(data_path)
    processed_netflix_data = process_netflix_data(netflix_data_all)
    netflix_data_with_series = identify_series_in_data(processed_netflix_data)
    return netflix_data_with_series


def identify_series_in_data(netflix_data):
    """
    This function tries to identify which are the series on the netflix data
     given. Then it creates an additional column indicating if the row is of a
     movie or a series (0 or 1 respectively).

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Pandas dataframe that must have the column title

    Returns
    -------
    netflix_data: pd.DataFrame
        Updated Pandas DataFrame with the new column 'is_serie'.

    """
    series_traits_list = [
        ": Season",
        ": Book",
        "(Episode ",
        " : Episode ",
        " : Part ",
        "(Chapter ",
        " : Chapter ",
        ": Temporada",
        ": Libro",
        "(Capítulo ",
        " : Capítulo ",
        " : Parte ",
        " : Episodio ",
        "(Episodio ",
    ]
    logging.info(f'Identifying if title contains any of: {series_traits_list}')
    series_trait_df = netflix_data.title == 'initialization of a false series'
    for series_trait in series_traits_list:
        series_trait_df += netflix_data.title.str.contains(
            series_trait,
            regex=False
        )

    netflix_data.loc[series_trait_df, 'is_serie'] = True
    netflix_data.loc[~series_trait_df, 'is_serie'] = False
    return netflix_data


def movie_and_series_information(df, profile=''):
    """
    This function gets information of the dataframe depending on the condition
    if it is a movie or if it is a series.
    For the movies part, the function 'merge_different_individual_start' is
     applied so that many registers of the same movie can be summarised into
     one row.
    For the series part, the function 'get_series_info' is applied to get all
    the information related to how the series was consumed.


    Parameters
    ----------
    df: pd.DataFrame
        Dataframe obtained from the netflix data, in this case the 'is_series'
         component must be in the dataframe (this comes from the
         identify_series_in_data function).
    profile: str
        profile name from which to filter.

    Returns
    -------
    information: tuple
        Tuple containing the four different dataframes:
            - movies: df of just the movies.
            - series: df of just the series.
            - movies_information: Df of the resumed information of the movies.
            - series_information: Df of the resumed information of the series.
    """
    tick = perf_counter()
    # We are filtering through a single profile.
    if profile:
        logging.info(f'Filtering netlfix data to only profile:{profile}.')
        data = df[df.profile_name == profile]
    else:
        data = df.copy()

    data.loc[data.is_serie == False, 'individual_start'] = data.title.map(
        data.title.value_counts()
    )

    logging.info(f'Analyzing only movies data.')
    movies = data[data.is_serie == False]
    movies_information = movies \
        .groupby('title') \
        .apply(merge_different_individual_start) \
        .drop_duplicates('title')

    # Data for series
    logging.info(f'Analyzing only series data.')
    series = data[data.is_serie == True]
    series_information = series\
        .groupby('new_title')\
        .apply(get_series_info)
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(
        f'Getting movies and series info took {time_it_took} seconds.'
    )
    information = movies, series, movies_information, series_information
    return information


def merge_different_individual_start(df):
    """
    This function must be executed on small dataframes where the same title is
    given. It measures different individual starts of movies and returns a
    list of the relevant information.

    Parameters
    ----------
    df: pd.DataFrame
        Small dataframe which is used to merge the information of certain
         features into list. Then those features are dropped.

        This df must contain at least the following columns available:
        - start_time
        - end_time
        - bookmark
        - duration
        - latest_bookmark

    Returns
    -------
    df_simplified: pd.DataFrame
        Dataframe with the new grouped columns and without other columns.

    """
    tick = perf_counter()
    df['start_time_list'] = [df.start_time.to_list()] * len(df)
    df['end_time_list'] = [df.end_time.to_list()] * len(df)
    df['bookmark_list'] = [df.bookmark.to_list()] * len(df)
    df['total_duration_seen'] = df.duration.sum() / 60
    drop_columns = [
        'start_time',
        'end_time',
        'duration',
        'bookmark',
        'latest_bookmark',
        'profile_name',
        'is_serie',
    ]
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
    df_simplified = df.reset_index()
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(
        f'Mergeing different individual start took {time_it_took} seconds.'
    )
    return df_simplified


def get_series_info(df):
    """
    This function gets relevant information of a dataframe regarding the
     nature of the series; that is: many chapters and many seasons.

    Parameters
    ----------
    df: pd.DataFrame
        This df must contain, at least, the following columns:
        - duration
        - end_time
        - start_time
        - title

    Returns
    -------
    results: pd.Series
        Pandas series with a resume on the relevant information of a netflix
         series.

    """
    tick = perf_counter()
    logging.info(
        f'Getting additional series information.'
    )
    total_duration = df.duration.sum() / 3600
    max_end_time = df.end_time.max()
    min_start_time = df.start_time.min()
    total_lapsed_time = (max_end_time - min_start_time).total_seconds() / 3600
    speed = total_duration / total_lapsed_time
    chapters = df.title.nunique()
    title = df.new_title.unique()[0]
    chapters_titles = df.title.to_list()
    all_start_times = df.start_time.apply(str).to_list()
    all_end_times = df.end_time.apply(str).to_list()

    chapter_speed = chapters / total_lapsed_time
    waiting_time_series = pd.Series(
        df.start_time.iloc[:-1].to_numpy() - df.end_time.iloc[1:].to_numpy()
    )
    waiting_time = waiting_time_series.apply(
        lambda x: x.total_seconds() / 3600
    )
    waiting_time_mean = waiting_time.mean()
    waiting_time_median = waiting_time.median()
    waiting_time_std = waiting_time.std()
    waiting_time_max = waiting_time.max()
    waiting_time_min = waiting_time.min()

    df['hour'] = df.start_time.apply(lambda x: x.hour + x.minute / 60)
    all_start_time_hours = df['hour'].to_list()

    results = {
        'new_title': title,
        'min_start_time': min_start_time,
        'max_end_time': max_end_time,
        'chapters_titles': chapters_titles,
        'all_start_times': all_start_times,
        'all_end_times': all_end_times,
        'all_start_time_hours': all_start_time_hours,
        'total_duration_hours': total_duration,
        'total_lapsed_hours': total_lapsed_time,
        'effective_seen_time': speed,
        'different_chapters_seen': chapters,
        'effective_seen_time_in_different_chapters': chapters * speed,
        'chapter_speed': chapter_speed,
        'waiting_time_mean': waiting_time_mean,
        'waiting_time_median': waiting_time_median,
        'waiting_time_std': waiting_time_std,
        'waiting_time_max': waiting_time_max,
        'waiting_time_min': waiting_time_min,
    }
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(
        f'Getting additional series info took {time_it_took} seconds.'
    )
    results = pd.Series(results)
    return results


def arrange_information_in_dict(general_ms, profile_ms):
    """
    This function gets the tuple general_ms and the tuple profile_ms and then
    arranges the data into a dictionary data structure of the following
    composition:

    info_series_movies = {
        {'general':
            {'movie':data_general_1,
            'series':data_general_2,
            'movie_info':data_general_3,
            'series_info':data_general_4}},
        {'profile1:
            {'movie':data_profile1_1,
            'series':data_profile1_2,
            'movie_info':data_profile1_3,
            'series_info':data_profile1_4}},
        ...
    }

    Parameters
    ----------
    general_ms: tuple
        Tuple containing dataframes that come from the application to
         movie_and_series_information of the general netflix_data file.
    profile_ms: tuple
        Tuple containing dataframes that come from the application to
         movie_and_series_information filtered by profile of the netflix_data
         file.


    Returns
    -------
    info_series_movies: dict
        Dictionary with the resumed information.

    """
    tick = perf_counter()
    logging.info(f'Arranging information.')
    info_series_movies = {'general': {}}
    sub_names = ['movie', 'series', 'movie_info', 'series_info']
    for index, sub_dataset in enumerate(sub_names):
        info_series_movies['general'][sub_dataset] = general_ms[
            index]
    for profile_num, profile in enumerate(profile_ms):
        info_series_movies[f'profile_{profile_num}'] = {}
        for index, sub_dataset in enumerate(sub_names):
            info_series_movies[f'profile_{profile_num}'][sub_dataset] = \
                profile_ms[profile_num][index]
    logging.info(f'General keys: {info_series_movies.keys()}')
    tock = perf_counter()
    time_it_took = tock-tick
    logging.info(
        f'Arranging information took {time_it_took} seconds.'
    )
    return info_series_movies


def save_dict_data(dict_data, path='./'):
    """
    Function to save each dataframe from the dictionary that comes from the
    arrange_information_in_dict function in a desired path.

    Parameters
    ----------
    dict_data: Dict
        Dictionary of the form provided in the arrange_information_in_dict
         function.
    path: str
        String that indicates the path where the file is going to be saved.

    Returns
    -------
    None

    """
    tick = perf_counter()
    logging.info('Saving info...')
    for key in dict_data.keys():
        for info in dict_data[key].keys():
            data = dict_data[key][info]
            save_data(
                data=data,
                path=path,
                name=f'{key}_{info}'
            )

    tock = perf_counter()
    time_it_took = tock-tick
    logging.info(f'Saving information took {time_it_took} seconds.')


def save_data(data, path='./', name='untitled'):
    """
    Function used to save data as a csv given the path and the file's name.

    Parameters
    ----------
    data: pd.DataFrame
        Desired dataframe to be saved.
    path: str
        Path where the csv must be located.
    name: str
        Name of the final file.

    Returns
    -------
    None

    """
    final_name = f'{path}/{name}.csv'
    data.to_csv(final_name, index=False)
    logging.info(f'File {name}.csv saved into {path}...')


def process():
    tick = perf_counter()
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    interim_data_path = os.path.join(data_path, 'interim')
    raw_data_path = os.path.join(data_path, 'raw')
    raw_folder = os.path.join(raw_data_path, 'netflix-report')
    interest_data_file = os.path.join(
        raw_folder,
        'CONTENT_INTERACTION/ViewingActivity.csv'
    )
    netflix_data = get_netflix_data(interest_data_file)
    general_ms_information = movie_and_series_information(netflix_data)
    profile_ms_information = netflix_data.groupby('profile_name').apply(
        movie_and_series_information
    )
    ms_information = arrange_information_in_dict(
        general_ms_information, profile_ms_information
    )
    save_data(data=netflix_data, path=interim_data_path, name='netflix_data')
    save_dict_data(dict_data=ms_information, path=interim_data_path)
    tock = perf_counter()
    time_it_took = tock-tick
    logging.info(f'Process took {time_it_took} seconds.')

if __name__ == "__main__":
    process()
