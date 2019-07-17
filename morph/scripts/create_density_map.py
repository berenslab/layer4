import pandas as pd
import numpy as np
import argparse
import sys
import os

from utils import NeuroFileLoader as loader
from utils import NeuronTree as nt
from utils_ import smooth_gaussian

DATASET_DESCRIPTION = "./data/dataset_description.csv"
FAULTY_CELLS = []#["HEC_2017 03 05 slice 3 cell 7.swc"]


def _project_data(dim, proj_axes, data):
    p_a = proj_axes.replace('x', "0").replace('y', "1").replace('z', "2")
    if dim == 2:
        if len(p_a) < 2:
            print('Invalid parameter setting: The passed projection axes {0} do not \
                     fit with the dimension of the projection {1}'.format(p_a, dim))
        else:
            indices = '012'
            for ix in range(len(p_a)):
                indices = indices.replace(p_a[ix], '')
            deleted_axis = int(indices)
            ax = [0, 1, 2]
            ax.remove(deleted_axis)
            result = data[:, ax]

    elif dim == 1:
        if len(p_a) > 1:
            print('Invalid parameter setting: The passed projection axes {0} do not \
                     fit with the dimension of the projection {1}'.format(p_a, dim))
        else:
            ax = int(p_a)
            result = data[:, ax]
    else:
        result = data

    return result


if __name__ == "__main__":

    # parse arguments from command line
    parser = argparse.ArgumentParser("Create density maps from neurons")
    parser.add_argument("filepath", type=str, nargs='+', help="path to file")
    parser.add_argument("-part", default='axon', choices=['axon', 'dendrite', 'full'],
                        help="neuron part that is used for computation.")

    parser.add_argument("-proj", default='xy', choices=['xy', 'xz', 'x', 'y', 'z'],
                        help='projection used in x, y and z coordinates.')

    args = parser.parse_args()

    # get range of coordinates to normalize
    extend = pd.read_csv("./data/cell_extend.csv",
                         engine='python').rename(columns={"Unnamed: 0": "coord"})
    r = dict(min=np.zeros((3,)), max=np.zeros((3,)))
    r['min'][0] = extend[extend["coord"] == args.part + "_width"]["min"]
    r['min'][1] = extend[extend["coord"] == args.part + "_height"]["min"]
    r['min'][2] = extend[extend["coord"] == args.part + "_depth"]['min']
    r['max'][0] = extend[extend["coord"] == args.part + "_width"]["max"]
    r['max'][1] = extend[extend["coord"] == args.part + "_height"]["max"]
    r['max'][2] = extend[extend["coord"] == args.part + "_depth"]['max']

    for file_path in args.filepath:
        # load Neuron Tree
        file_name = file_path.split("/")[-1]
        save_path = "./data/density_maps/aligned_globally/%s/%s/" % (args.part,
                                                                                                        args.proj)
        if file_name not in FAULTY_CELLS or not os.path.exists(save_path + "%s.txt" % file_name[:-4]):
            print('Calculating density map for %s \n part: %s ' % (file_path, args.part))
            path = file_path[: len(file_path) - len(file_name)]
            swc = loader.load_swc_files(path=path, file_name=file_name, cell_id=0)
            N = nt.NeuronTree(swc=swc)

            # get point cloud sampled along branches in dist=100nm
            dist = 0.1

            if args.part == 'axon':
                pc = nt.NeuronTree.resample_nodes(N.get_axonal_tree().get_graph(), dist)
            elif args.part == 'dendrite':
                pc = nt.NeuronTree.resample_nodes(N.get_dendritic_tree().get_graph(), dist)
            elif args.part == 'full':
                pc = nt.NeuronTree.resample_nodes(N.get_graph(), dist)

            # normalize point cloud
            pc = (pc - r['min']) / (r['max'] - r['min'])
            dims = len(args.proj)

            # project data
            data = _project_data(dims, args.proj, pc)

            # create histogram
            range_ = [[-.1, 1.1]] * dims
            H, edges = np.histogramdd(data, bins=(100,) *dims, range=range_, normed=True)

            # smooth with a gaussian filter
            H = smooth_gaussian(H, dim=dims, sigma=2)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savetxt(save_path + "%s.txt" % file_name[:-4], H.reshape(-1))

