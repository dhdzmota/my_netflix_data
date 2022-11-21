import os


pipeline = [
    'data/initial_data_unzip_extraction',
    'data/movies_and_series',
    'visualization/create_visualizations',
]


def run_pipeline():
    """
    Function that finds the most relevant python commands and runs them.
    Returns
    -------
    None
    """
    general_path = os.path.join(os.path.dirname(__file__), 'src')
    for script_name in pipeline:
        script = os.path.join(general_path, f'{script_name}.py')
        command = f'python {script}'
        os.system(command)


if __name__ == '__main__':
    run_pipeline()
