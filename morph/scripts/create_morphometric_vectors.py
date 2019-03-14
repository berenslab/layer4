import pandas as pd
import numpy as np
import argparse
import sys
import os

sys.path.append("/mnt/remote_home/Projects/BC_morphologies/git/sophie/")
sys.setrecursionlimit(1000)

from utils import NeuroFileLoader as loader
from utils import NeuronTree as nt
from queue import Queue
from threading import Thread
import networkx as nx
import copy

DATASET_DESCRIPTION = "../data/dataset_description.csv"
INPUT_FOLDER = "../data/nt/"
FAULTY_CELLS = []


def worker():
    while True:
        data = q.get()
        if data is None:  # EOF?
            return
        file_path, part = data

        file_name = file_path.split("/")[-1]
        print('Calculating morphometric statistics for %s \n part: %s ' % (file_path, part))
        path = file_path[: - len(file_name)]
        swc = loader.load_swc_files(path=path, file_name=file_name, cell_id=0)
        N = nt.NeuronTree(swc=swc)

        if args.part == 'axon':
            T = N.get_axonal_tree()
        elif args.part == 'dendrite':
            T = N.get_dendritic_tree()
        else:
            T = N
        z = dict()
        z['name'] = file_name
        z['type'] = file_name.split("_")[0]
        z['branch_points'] = T.get_branchpoints().size
        extend = T.get_extend()

        z['width'] = extend[0]
        z['depth'] = extend[1]
        z['height'] = extend[2]

        z['tips'] = T.get_tips().size

        z['stems'] = len(T.edges(1))

        z['total_length'] = np.sum(list(nx.get_edge_attributes(T.get_graph(), 'euclidean_dist').values()))

        z['max_path_dist_to_soma'] = np.max(T.get_distance_dist()[1])
        z['max_branch_order'] = np.max(list(T.get_branch_order().values()))

        path_angles = []
        for p1 in T.get_path_angles().items():
            if p1[1].values():
                path_angles += list(list(p1[1].values())[0].values())

        z['max_path_angle'] = np.max(path_angles)
        z['min_path_angle'] = np.min(path_angles)
        z['mean_path_angle'] = np.mean(path_angles)

        R = T.get_mst()

        z['max_segment_length'] = np.max(list(R.get_segment_length().values()))

        tortuosity = [e[2]['path_length']/e[2]['euclidean_dist'] for e in R.edges(data=True)]

        z['max_tortuosity'] = np.max(tortuosity)
        z['min_tortuosity'] = np.min(tortuosity)
        z['mean_tortuosity'] = np.mean(tortuosity)

        branch_angles = R.get_branch_angles()
        z['max_branch_angle'] = np.max(branch_angles)
        z['min_branch_angle'] = np.min(branch_angles)
        z['mean_branch_angle'] = np.mean(branch_angles)

        result[file_path] = z # store it


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser("Create vector of morphometric statistics from neurons")
    parser.add_argument("filename", default='',nargs='+', type=str, help="name of file. Is supposed to be in INPUT FOLDER")
    parser.add_argument("-part", default='axon', choices=['axon', 'dendrite', 'full'],
                        help="neuron part that is used for computation.")

    args = parser.parse_args()

    swcs = []

    save_path = "../data/morphometrics/%s/" % args.part
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        morphometry_data = pd.read_csv(save_path + 'morphometrics.csv')
        existing_files = np.unique(morphometry_data['name'])
    except FileNotFoundError:
        morphometry_data = pd.DataFrame()
        existing_files = []

    if args.filename:

        inputs = args.filename
    else:
        inputs = INPUT_FOLDER + os.listdir(INPUT_FOLDER)

    q = Queue()
    result = {}  # used to store the results

    for file_path in inputs:

        file_name = file_path.split("/")[-1]
        if file_name not in FAULTY_CELLS or file_name in existing_files:

            q.put((file_path, args.part))

    threads = [Thread(target=worker) for _i in range(8)]
    for thread in threads:
        thread.start()
        q.put(None)  # one EOF marker for each thread

    for x in threads:
        x.join()

    for z in result.values():
        morphometry_data = morphometry_data.append(pd.DataFrame(z, index=['name']),sort=True)
    morphometry_data.to_csv(save_path + 'morphometrics.csv')





