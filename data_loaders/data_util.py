import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset


data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
# Usage: name = data_sources[1], idx = data_sources.index('ijrr')


def concatenate_subfolders(data_file, dataset, dataset_kwargs):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """
    if os.path.isdir(data_file):
        subfolders = [os.path.join(data_file, s) for s in os.listdir(data_file)]
    elif os.path.isfile(data_file):
        subfolders = pd.read_csv(data_file, header=None).values.flatten().tolist()
    else:
        raise Exception('{} must be data_file.txt or base/folder'.format(data_file))
    print('Found {} samples in {}'.format(len(subfolders), data_file))
    datasets = []
    for subfolder in subfolders:
        dataset_kwargs['item_kwargs'].update({'base_folder': subfolder})
        datasets.append(dataset(**dataset_kwargs))
    return ConcatDataset(datasets)


def concatenate_datasets(data_file, dataset_type, dataset_kwargs=None):
    """
    Generates a dataset for each cti_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to CTI h5 files.
                      Each file is expected to have a sequence of frame_{:09d}
    :param dataset_type: Pointer to dataset class
    :param dataset_kwargs: Dataset keyword arguments
    :return ConcatDataset: concatenated dataset of all cti_paths in data_file
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    cti_paths = pd.read_csv(data_file, header=None).values.flatten().tolist()
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for cti_path in tqdm(cti_paths):
        dataset_kwargs['dataset_kwargs'].update({'h5_path': cti_path})
        dataset_list.append(dataset_type(**dataset_kwargs))
    return ConcatDataset(dataset_list)


def concatenate_memmap_datasets(data_file, dataset_type, dataset_kwargs):
    """
    Generates a dataset for each memmap_path specified in data_file and concatenates the datasets.
    :param data_file: A file containing a list of paths to memmap root dirs.
    :param dataset_type: Pointer to dataset class
    :param dataset_kwargs: Dataset keyword arguments
    :return ConcatDataset: concatenated dataset of all memmap_paths in data_file
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    memmap_paths = pd.read_csv(data_file, header=None).values.flatten().tolist()
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for memmap_path in tqdm(memmap_paths):
        dataset_kwargs['dataset_kwargs'].update({'root': memmap_path})
        dataset_list.append(dataset_type(**dataset_kwargs))
    return ConcatDataset(dataset_list)
