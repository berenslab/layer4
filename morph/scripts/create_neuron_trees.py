import sys
import argparse
import pandas as pd
import os

from utils import NeuroFileLoader as loader
from utils import NeuronTree as nt
from utils_ import get_standardized_swc
from multiprocessing import Pool


PATH_TO_DATASET_DESCRIPTION = "/mnt/remote_home/Projects/V1 Layer 4/data/dataset_description.csv"
SAVE_FOLDER = "/mnt/remote_home/Projects/V1 Layer 4/data/nt/"


def create_neuron_tree(input_file):
    print(input_file)
    file_name = input_file.split("/")[-1]
    cell_type = input_file.split("/")[-2]


    path = input_file[:-len(file_name)]
    swc = loader.load_swc_files(path=path, file_name=file_name, cell_id=0)

    # switch y and z since y corresponds to cortical depth
    swc = swc.rename(columns={'y': 'z', 'z': 'y'})
    # rotate x,y coordinates into their frame of maximal extend  and soma center for standardization
    rotated_swc = get_standardized_swc(swc)

    N = nt.NeuronTree(swc=rotated_swc)

    N.write_to_swc(file_name=cell_type + "_" + file_name[:-4], ext="",
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

