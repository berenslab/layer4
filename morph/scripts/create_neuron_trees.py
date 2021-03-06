import sys
import argparse
import pandas as pd
import os

from utils import NeuroFileLoader as loader
from utils import NeuronTree as nt
from utils_ import get_standardized_swc
from multiprocessing import Pool


PATH_TO_DATASET_DESCRIPTION = "./data/dataset_description.csv"
SAVE_FOLDER = "./data/nt/"


def create_neuron_tree(input_file):
    print(input_file)
    file_name = input_file.split("/")[-1]
    cell_type = input_file.split("/")[-2]


    path = input_file[:-len(file_name)]
    swc = loader.load_swc_files(path=path, file_name=file_name, cell_id=0)

    # switch y and z since y corresponds to cortical depth
    swc = swc.rename(columns={'y': 'z', 'z': 'y'})
    # reduce soma to one point and soma center for standardization
    standardized_swc = get_standardized_swc(swc)

    N = nt.NeuronTree(swc=standardized_swc)
    N = N.resample_tree(dist=1)
    N = N.smooth_neurites(dim=1, window_size=21)
    N.write_to_swc(file_name=cell_type + "_" + file_name[:-4],
                   path=SAVE_FOLDER)


if __name__ == "__main__":

    # get description to translate input file path into cell number
    dataset = pd.read_csv(PATH_TO_DATASET_DESCRIPTION)

    parser = argparse.ArgumentParser(description='Create neuron trees.')
    parser.add_argument("--force_all", default=False, help='forces to recalculate all data')
    args = parser.parse_args()

    swc_paths = []
    for n, row in dataset.iterrows():
        file_path = SAVE_FOLDER + row['type']+"_" + row['swc'][:-4] + ".swc"
        if not os.path.exists(file_path) or args.force_all:
            swc_paths.append(row['path'] + row['swc'])

    with Pool(4) as p:
        p.map(create_neuron_tree, swc_paths)

