import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import warnings

from ast import literal_eval
from PyPDF2 import PdfMerger
from sklearn.cluster import DBSCAN

from src.data.fetch_information import (
    get_processed_netflix_data,
    get_general_sorted_data,
)
from src.visualization.utils import (
    get_pivoted_data,
    clean_text,
    create_calendar_pivot_table,
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
    warnings.filterwarnings("ignore")
    colormap = mpl.colors.LinearSegmentedColormap.from_list(
        "", ['red', 'black']
    ).resampled(100)
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
    plt.savefig(
        f'{image_path}img1_netflix_proporcion_fecha_perfil.pdf',
        bbox_inches='tight'
    )
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
    plt.savefig(
        f'{image_path}img2_series__{clean_title_text}.pdf',
        bbox_inches='tight'
    )
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
    additional_string = ''
    if filter_profile_name:
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

    plt.savefig(save_name,bbox_inches='tight')
    plt.close()


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
    colormap = initialize_configuration()
    general_path = os.path.join(os.getcwd())
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


if __name__ == "__main__":
    process()
