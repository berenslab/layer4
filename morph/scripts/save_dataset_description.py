import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # parse in input and output if it is specified
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = '/mnt/berens_data/data/anatomy/V1 L4 Scala/swc/'

    try:
        save_path = sys.argv[2]
    except IndexError:
        save_path = "/mnt/remote_home/Projects/V1 Layer 4/data/dataset_description.csv"

    # delete the file if it exists previously
    if os.path.exists(save_path):
        os.remove(save_path)

    swc_types = os.listdir(data_path)
    swcs = []
    folder_names = []
    # get cell types from folder names and collect all swc files in those folders
    for t in swc_types:
        all_files = list(os.walk(data_path + t))[0][2]
        swc_files = [f for f in all_files if f[-3:] == 'SWC']
        folder_names += ([t] * len(swc_files))
        swcs += swc_files

    # convert cell type names into akronyms
    cell_types = np.array(folder_names)

    file_paths = [data_path+f+"/" for f in folder_names]
    # create Data frame that contains swc file names and their type 
    data = pd.DataFrame(np.vstack((np.array(swcs),cell_types, file_paths)).T, columns=['swc', 'type', 'path'])

    data.to_csv(save_path) # save file
