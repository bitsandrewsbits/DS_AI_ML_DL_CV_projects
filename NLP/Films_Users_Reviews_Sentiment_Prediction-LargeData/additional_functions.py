# additional functions
import re
import os

def get_minidatasets_filenames_in_data_dir():
    minidatasets_filenames = []
    data_dir_files = os.listdir('data')
    for data_dir_file in data_dir_files:
        if re.match(r'minidataset_[1-9][0-9]?.csv', data_dir_file):
            minidatasets_filenames.append(data_dir_file)
    return minidatasets_filenames

def remove_old_prepared_result_minidatasets():
    data_dir_files = os.listdir('data')
    for file in data_dir_files:
        if re.match(r'result_minidataset_.*', file):
            os.remove(f'data/{file}')
        elif re.match(r'prepared_minidataset_.*', file):
            os.remove(f'data/{file}')
