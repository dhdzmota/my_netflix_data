import datetime
import glob
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

from ast import literal_eval
from PIL import Image
from PyPDF2 import PdfMerger
from sklearn.cluster import DBSCAN
from time import perf_counter

from src.data.fetch_information import (
    get_processed_netflix_data,
    get_general_sorted_data,
)
from src.visualization.utils import (
    get_pivoted_data,
    clean_text,
    create_calendar_pivot_table,
    colorfunc,
    create_folder,
    delete_folder
)


def initialize_configuration():
    """
    This function establishes the initial configuration:
     - Ignore warnings.
     - Use a hand-crafted colormap.

    Returns
    -------
    colormap: matplotlib.colors.LinearSegmentedColormap
        The desired colormap resampled (given a number from 0 to 1 a color of
         the colormap will return a corresponding value mapped into 100
         different tones).

    """
    logging.info('Initializing configuration')
    warnings.filterwarnings("ignore")
    logging.info('Ignoring warinigs.')
    colorlist = ['red', 'black']
    colormap = mpl.colors.LinearSegmentedColormap.from_list(
        "", colorlist
    ).resampled(100)
    logging.info(f'Colormap uses colrlist: {colorlist}.')
    return colormap


def get_stacked_profile_duration(netflix_data, image_path='./', cmap=None):
    """
    This functions generates a stacked plot over time with the proportion of
     time spent by each profile of the same account. The values are grouped
     each two months to make it more user-friendly when analyzing the data.

    In this case, at any given time, the sum of all the profile values must be
     equal to 1.

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Processed netflix data obtained from the get_processed_netflix_data
         function.
    image_path: str
        String of the path where the images will be saved in.
    cmap: matplotlib.colors.LinearSegmentedColormap
        Desired colormap.

    Returns
    -------
    None
    """
    logging.info('Getting stacked profile duration file.')
    pivoted_data = get_pivoted_data(netflix_data)
    pivoted_data_filtered = pivoted_data[
        [col for col in pivoted_data.columns if
         not col.startswith('_') and 'total' not in col]
    ]
    pivoted_data_filtered.reset_index(inplace=True)
    pivoted_data_filtered.set_index('start_time', inplace=True)

    pivoted_data_filtered.plot.area(
        stacked=True,
        linewidth=0.1,
        figsize=(13, 8),
        cmap=cmap,
        legend=True
    )
    plt.ylabel('Duración (horas)')
    plt.xlabel('Fecha')
    title_str = 'Duración de tiempo visto en Netflix para cada perfil'
    plt.title(title_str)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc="center right")
    plt.grid(linestyle='--')
    plt.savefig(
        f'{image_path}img0_netflix_duracion_fecha_perfil.pdf',
        bbox_inches='tight'
    )
    plt.close()


def get_stacked_profile_proportion(netflix_data, image_path='./', cmap=None):
    """
    This functions generates a stacked plot over time with the duration of time
     spent by each profile of the same account. The values are grouped each
     two months to make it more user-friendly when analyzing the data.

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Processed netflix data obtained from the get_processed_netflix_data
         function.
    image_path: str
        String of the path where the images will be saved in.
    cmap: matplotlib.colors.LinearSegmentedColormap
        Desired colormap.

    Returns
    -------
    None
    """
    logging.info('Getting stacked profile proportion file.')
    pivoted_data = get_pivoted_data(netflix_data)
    pivoted_data_filtered = pivoted_data[
        [col for col in pivoted_data.columns if
         col.startswith('_') and '_t' not in col]
    ]
    columns_rename = {col: col[1:] for col in pivoted_data_filtered.columns}
    pivoted_data_filtered.rename(
        columns=columns_rename,
        inplace=True,
    )
    pivoted_data_filtered.reset_index(inplace=True)
    pivoted_data_filtered.set_index('start_time', inplace=True)
    pivoted_data_filtered.plot.area(
        stacked=True,
        linewidth=0.1,
        figsize=(13, 8),
        cmap=cmap,
        legend=True
    )
    plt.ylabel('Proporción')
    plt.xlabel('Fecha')
    plt.ylim(0, 1)
    title_str = 'Proporción de tiempo en netflix para cada perfil.'
    plt.title(title_str)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc="center right")
    save_name = f'{image_path}img1_netflix_proporcion_fecha_perfil.pdf'
    plt.savefig(
        save_name,
        bbox_inches='tight'
    )
    logging.info(f'Saving plot into {save_name}')
    plt.close()


def plot_series_time(series_data_row, image_path='./', cmap=None):
    """
    This function plots a series over time (just like a time series, no pun
     intended) where the x_axis is the starting point and the y axis is the
     time of the day the series was watched. It also uses the colormap to
     identify if the data points are too sepparated or too close together. This
     is done through an Unsupervised machine learning model called DBSCAN.
     The sepparation of clusters is 48 hours (2 days).

    Parameters
    ----------
    series_data_row: pd.DataFrame
        Data frame that comes from the function get_general_sorted_data. In
         this case, it is specifically for series informaiton (not movies nor
         general netflix data).
    image_path: str
        String of the path where the images will be saved in.
    cmap: matplotlib.colors.LinearSegmentedColormap
        Desired colormap.

    Returns
    -------
    None
    """
    logging.info('Getting plot of series over time.')
    series_title = series_data_row.new_title
    start_times = pd.to_datetime(literal_eval(series_data_row.all_start_times))
    hour = literal_eval(series_data_row.all_start_time_hours)
    now_time_difference = pd.Series(
        start_times - datetime.datetime.now()
    ).apply(lambda x: x.days).to_numpy().reshape(-1, 1)
    clustering = DBSCAN(
        min_samples=1,
        eps=48
    ).fit_predict(now_time_difference)

    # Create figure and plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plt.scatter(start_times, hour, c=clustering, cmap=cmap)
    plt.plot(start_times, hour, linestyle='--', alpha=0.5, color='k')
    plt.ylabel('Hora del día')
    plt.xlabel('Fecha')
    plt.title(series_title)
    plt.yticks([0, 4, 8, 12, 16, 20, 24])
    plt.ylim(-0.5, 24.5)
    plt.grid(linestyle='--')
    ax.patch.set_facecolor('gainsboro')
    clean_title_text = clean_text(series_title)
    save_name = f'{image_path}img2_series__{clean_title_text}.pdf'
    plt.savefig(
        save_name,
        bbox_inches='tight'
    )
    logging.info(f'Saving plot into {save_name}')

    plt.close()


def generate_calendarlike_plot(netflix_data, image_path='./', cmap=None,
                               filter_profile_name=''):
    """
    This function produces a calendar-like plot with the pivot table generated
     inside with the function create_calendar_pivot_table.

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Processed netflix data obtained from the get_processed_netflix_data
         function.
    image_path: str
        String of the path where the images will be saved in.
    cmap: matplotlib.colors.LinearSegmentedColormap
        Desired colormap.
    filter_profile_name: str
        The netflix_data comes with a column called profile_name, thus it can
        be filtered.

    Returns
    -------
    None

    """
    logging.info('Getting calendarlike plot.')

    additional_string = ''
    if filter_profile_name:
        logging.info(f'Plot for just profile {filter_profile_name}')
        netflix_data = netflix_data[
            netflix_data.profile_name == filter_profile_name
        ]
        additional_string = f' para el perfil: {filter_profile_name}'
    calendarized = create_calendar_pivot_table(netflix_data)
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        calendarized,
        annot=True,
        linewidth=.5,
        cmap=cmap
    )
    title = 'Horas totales vistas de Netflix cada mes'
    if additional_string:
        title += additional_string
    plt.title(title)
    plt.xlabel('Número de mes')
    plt.ylabel('Año')
    save_name = f'{image_path}img3_netflix_horas_mes_anio.pdf'
    if additional_string:
        save_name = f'{image_path}img3_netflix_horas_mes_anio' \
                    f'__{filter_profile_name}.pdf'
    logging.info(f'Saving plot into {save_name}')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

    nan_calendarized = calendarized.copy()
    for col in nan_calendarized.columns:
        nan_calendarized[col] = np.nan
    animation_path = os.path.join(image_path, 'animations')
    temporary_table_path = os.path.join(animation_path, 'temporary_tables')

    figure_name = 'calendar_tmp'
    if additional_string:
        figure_name += f'__{filter_profile_name}'

    figures_path_tmp = os.path.join(temporary_table_path, figure_name)
    create_folder(animation_path)
    create_folder(temporary_table_path)
    create_folder(figures_path_tmp)
    for row in nan_calendarized.index:
        for col in nan_calendarized.columns:
            if not np.isnan(calendarized[col].loc[row]):
                nan_calendarized[col].loc[row] = calendarized[col].loc[row]
                plt.figure(figsize=(20, 10))
                sns.heatmap(
                    nan_calendarized,
                    annot=True,
                    linewidth=.5,
                    vmin=calendarized.min().min(),
                    vmax=calendarized.max().max(),
                    cmap=cmap,
                )
                title = 'Horas totales vistas de Netflix cada mes'
                if additional_string:
                    title += additional_string
                plt.title(title)
                plt.xlabel('Número de mes')
                plt.ylabel('Año')

                save_fig_name = f'{figures_path_tmp}/' \
                                f'figure_{row}_{str(col).zfill(2)}'
                if additional_string:
                    save_fig_name += f'__{filter_profile_name}'
                plt.savefig(
                    f'{save_fig_name}.png', bbox_inches='tight'
                )
                plt.close()
    gifname = 'heatmap'
    if additional_string:
        gifname += f'__{filter_profile_name}'
    generate_gif_with_extra_info(
        figures_path_tmp,
        animation_path,
        name=gifname
    )
    delete_folder(figures_path_tmp)


def animate_df_total_time(netflix_data, days=5):
    ''' This is a helper function to animate the total_time by filtering over
    time the dataframe

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Data from netflix.
    days: int
        The number of days that are taken into account when iterating for a
        slight increase on the registers.

    Returns
    -------
    dataframe_over_time: pd.DataFrame
        A dataframe with the desired time condition so it can be properly
        animated.
    '''
    logging.info(f'Using a time delta of {days} days.')
    time_delta = datetime.timedelta(days=days)
    upper_time_border = netflix_data.start_time.min() + time_delta
    time_condition = netflix_data.start_time <= upper_time_border
    logging.info('Filtering data with the time condition.')
    dataframe_over_time = netflix_data[time_condition]
    logging.info('Getting last value (since it is a cumulative sum).')
    new_rows = dataframe_over_time.groupby('profile_name').max().reset_index()
    new_rows.start_time = netflix_data.start_time.min() + time_delta
    logging.info(
        'Appending the last value to the dataframe for a proper animation.'
    )
    dataframe_over_time = dataframe_over_time.append(new_rows)
    return dataframe_over_time


def animate_total_time(netflix_data, colormap, image_path='./', dayspeed=5):
    '''
    This function gets the netflix data and creates a gif on the total amount
    of time a profile has been watching netflix over the accounts' history.

    Parameters
    ----------
    netflix_data: pd.DataFrame
        Dataframe of netflix data.
    colormap: matplotlib.colors.LinearSegmentedColormap
        The desired colormap resampled (given a number from 0 to 1 a color of
         the colormap will return a corresponding value mapped into 100
         different tones).
    image_path: str
        The usual path that is used to save images.
    dayspeed: int
        Steps of which to make the number of days between the maximum date and
         the minimum date.

    Returns
    -------
    None

    '''
    netflix_data = netflix_data.sort_values('start_time')
    logging.info('Obtaining the cumulative sum over time for each profile.')
    netflix_data['duration_cumsum'] = netflix_data.groupby(
        'profile_name').duration.cumsum()

    total_profiles = netflix_data.profile_name.unique()
    logging.info(
        f'Resample the dataframe to the length of the '
        f'profiles: {total_profiles}'
    )
    cmp = colormap.resampled(len(total_profiles))
    profile_dict = {}
    logging.info('Getting a color for each profile.')
    for i, profile in enumerate(total_profiles):
        profile_dict[profile] = colorfunc(total_profiles, cmp, i)

    total_days = (
            netflix_data.end_time.max() - netflix_data.start_time.min()
    ).days
    logging.info(f'Total days between min and max time in data: {total_days}')
    logging.info(f'Creating a list from 0 to {total_days} '
                 f'in steps of {dayspeed}')

    days_passing = list(range(0, total_days, dayspeed))
    animation_path = os.path.join(image_path, 'animations')
    logging.info(f'Animation path:{animation_path}')
    temporary_table_path = os.path.join(
        animation_path, 'temporary_tables'
    )
    figures_path_tmp = os.path.join(
        temporary_table_path, 'cumsum_overtime_tmp')

    logging.info('Creating folders')
    create_folder(animation_path)
    create_folder(temporary_table_path)
    create_folder(figures_path_tmp)

    for day in days_passing:
        fig = plt.figure()
        ax = plt.gca()
        nf_temporal = animate_df_total_time(netflix_data, day)
        logging.info(f'Graph each profile in day: {day}')
        for profile in sorted(profile_dict.keys()):
            nf_temporal_p = nf_temporal[nf_temporal.profile_name == profile]
            new_row = nf_temporal[
                nf_temporal.start_time == nf_temporal.start_time.max()].copy()
            new_row['duration_cumsum'] = nf_temporal_p['duration_cumsum'].max()
            nf_temporal_p_actualized = nf_temporal_p.append(new_row)
            nf_temporal_p_actualized.plot(
                x='start_time',
                y='duration_cumsum',
                ax=ax,
                label=profile,
                color=profile_dict[profile],
            )
            plt.scatter(
                [nf_temporal_p_actualized.start_time.max()],
                [nf_temporal_p_actualized.duration_cumsum.max()],
                color=profile_dict[profile],
            )
        plt.title('Tiempo total viendo Netflix')
        plt.ylabel('Tiempo (horas)')
        plt.xlabel('Fecha')
        number = str(day).zfill(4)
        plt.savefig(f'{figures_path_tmp}/figure_{number}.png')
        plt.close()

    logging.info('Generating GIF...')
    generate_gif(figures_path_tmp, animation_path)
    delete_folder(figures_path_tmp)


def generate_gif(figures_path_tmp, animation_path, name='image'):
    """
    This function creates a gif from a bunch of saved images (in a folder).

    Parameters
    ----------
    figures_path_tmp: str
        path to the figures.
    animation_path: str
        path to the animation (where should the resulting file be)
    name:
        name of the animation.

    Returns
    -------
    None

    """
    fp_in = f"{figures_path_tmp}/figure_*.png"
    fp_out = f"{animation_path}/{name}.gif"
    logging.info(f'Reading files from path: {figures_path_tmp}')
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)
    logging.info(f'Saving Gif in {fp_out}')
    img.save(
        fp=fp_out,
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=15,
        loop=0
    )


def generate_gif_with_extra_info(
    figures_path_tmp, animation_path, name='image',
):
    """
    This function creates a gif with extra information; it takes into account
    the amount of images and assigns a different duration to each transitions
    in order to create a better gif.

    Parameters
    ----------
    figures_path_tmp: str
        path to the figures.
    animation_path: str
        path to the animation (where should the resulting file be)
    name:
        name of the animation.

    Returns
    -------
    None

    """
    fp_in = f"{figures_path_tmp}/figure_*.png"
    fp_out = f"{animation_path}/{name}.gif"
    logging.info(f'Reading files from path: {figures_path_tmp}')

    dirlist = os.listdir(figures_path_tmp)
    total_imgs = len(dirlist)
    logging.info(f'Total files: {total_imgs}')

    max_time = 100
    min_time = 20
    alpha = (max_time / (total_imgs) ** 2)

    actual_max_time = max_time + min_time
    time_vec_x = np.linspace(-total_imgs, 0, total_imgs)
    time_vec = alpha * (time_vec_x) ** 2 + min_time
    time_list = list(time_vec)
    time_list[-1] = 1000

    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)
    logging.info(f'Saving gif in path: {fp_out}')
    img.save(
        fp=fp_out,
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=time_list,
        loop=0,
    )


def generate_report(image_path, report_path):
    """
    This function grabs all the .pdf files that are in the image_path location
    and then merges them into a single .pdf file saved in the following file:
    {report_path}/report.pdf.

    Parameters
    ----------
    image_path: str
        Os path to the image location.
    report_path: str
        Os path to the report file location.
    Returns
    -------
    None
    """
    merger = PdfMerger()
    for pdf_file in os.listdir(image_path):
        if pdf_file.endswith('.pdf'):
            merger.append(f'{image_path}{pdf_file}')

    merger.write(f"{report_path}report.pdf")
    merger.close()


def process():
    """
    Main process function.

    Returns
    -------
    None
    """
    tick = perf_counter()
    colormap = initialize_configuration()
    general_path = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(general_path, 'data')
    interim_data_path = os.path.join(data_path, 'interim')
    interest_data_file = os.path.join(
        interim_data_path,
        'netflix_data.csv'
    )
    interest_series_file = os.path.join(
        interim_data_path,
        'profile_0_series_info.csv'
    )
    report_path = os.path.join(general_path, 'reports/')
    images_data_path = os.path.join(report_path, 'figures/')

    netflix_data = get_processed_netflix_data(interest_data_file)
    series_data = get_general_sorted_data(
        interest_series_file,
        sorted_by='total_duration_hours',
        limit_rows=30
    )

    get_stacked_profile_duration(netflix_data, images_data_path, colormap)
    get_stacked_profile_proportion(netflix_data, images_data_path, colormap)

    animate_total_time(
        netflix_data,
        colormap=colormap,
        image_path=images_data_path
    )
    generate_calendarlike_plot(
        netflix_data=netflix_data,
        image_path=images_data_path,
        cmap=colormap,
    )
    for profile in netflix_data.profile_name.unique():
        generate_calendarlike_plot(
            netflix_data=netflix_data,
            image_path=images_data_path,
            cmap=colormap,
            filter_profile_name=profile,
        )
    series_data.sort_values('total_duration_hours', ascending=False)\
        .apply(
            plot_series_time,
            image_path=images_data_path,
            cmap=colormap,
            axis=1,
        )
    generate_report(images_data_path, report_path)
    tock = perf_counter()
    time_it_took = tock - tick
    logging.info(f'Extracting files took {time_it_took} seconds.')


if __name__ == "__main__":
    process()
