import pandas as pd
import numpy as np
import h5py

def load_t2_files(cell_id,
                  path='/gpfs01/berens/data/data/anatomy/BC_morphologies/nc_morphologies_vaa3d/'):
    # load morphologies formatted for NeuronC
    morph = pd.read_csv(path + 'morph.cell0' + str(cell_id) + '.t2', delim_whitespace=True, comment='#',
                        names=['node', 'parent', 'dia', 'xbio', 'ybio', 'zbio', 'region', 'dendr'])

    return morph


def load_swc_files(cell_id,
                   path='/gpfs01/berens/data/data/anatomy/BC_morphologies/swc_vaa3d/',
                   file_name=None):
    # read swc files as created from Vaa3D
    if file_name:
        swc = pd.read_csv(path + file_name, delim_whitespace=True, comment='#',
                          names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False)
    else:
        swc = pd.read_csv(path + 'cell0' + str(cell_id) + '.swc', delim_whitespace=True, comment='#',
                          names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False)
    return swc


def load_volume_data(cell_id, path='/gpfs01/berens/data/data/anatomy/BC_morphologies/volume_data/'):
    data = np.empty((0, 3))
    for x in range(12, 18):  # (12,23)
        try:
            with h5py.File(path + 'data_OPL_BC/cell' + str(cell_id).zfill(4) + 'x' + str(x).zfill(4) + '.hdf5',
                           'r') as f:
                data = np.concatenate((data, f['/cell'][:].reshape(-1, 3)), axis=0)
        except OSError:
            continue
    for x in range(18, 33):
        try:
            with h5py.File(path + 'data_INL_somata_fine/cell' + str(cell_id).zfill(4) + 'x' + str(x).zfill(4) + '.hdf5',
                           'r') as f:
                data = np.concatenate((data, f['/cell'][:].reshape(-1, 3)), axis=0)
        except OSError:
            continue
    for x in range(33, 57):  # (31,57)
        try:
            with h5py.File(path + 'data_IPL/cell' + str(float(cell_id)).zfill(4) + 'x' + str(x).zfill(4) + '.hdf5',
                           'r') as f:
                data = np.concatenate((data, f['/cell'][:].reshape(-1, 3)), axis=0)
        except OSError:
            continue
    return data



