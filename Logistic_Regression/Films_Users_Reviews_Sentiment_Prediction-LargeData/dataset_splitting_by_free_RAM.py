# this is program for analyzing your free RAM and
# compute optimal size of minidatasets for reading, changing operations.
# Minidatasets will be writting into disk into dir where original large dataset exists.

import psutil

class Dataset_Splitting_by_free_RAM:
    def __init__(self, csv_dataset_path: str):
        self.csv_dataset_path = csv_dataset_path
        self.free_memory_for_minidataset = self.get_optimal_free_RAM()

    def get_optimal_free_RAM(self):
        virt_memory = psutil.virtual_memory()
        free_RAM_in_GB = virt_memory.available / (1024 * 1024 * 1024)
        optimal_free_RAM_for_minidataset = int(free_RAM_in_GB) - 1
        if optimal_free_RAM_for_minidataset > 0:
            print('Optimal RAM for minidataset')
            print(f'for reading, changing operations = {optimal_free_RAM_for_minidataset}GB')
            return optimal_free_RAM_for_minidataset
        else:
            print('Your free RAM very low. If you want to continue, release memory.')
            return False

if __name__ == '__main__':
    dataset_splitter = Dataset_Splitting_by_free_RAM('data/user_reviews.csv')
