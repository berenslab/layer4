import numpy as np
import pandas as pd
import os
import copy

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.signal import convolve2d, gaussian
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm, gaussian_kde


def get_standardized_swc(swc, scaling=1., soma_radius=None, pca_rot=True):
    """

    :param swc:
    :param scaling:
    :param soma_radius:
    :param pca_rot:
    :param soma_center:
    :return:
    """

    swc.update(swc['x'] / scaling)
    swc.update(swc['y'] / scaling)
    swc.update(swc['z'] / scaling)
    swc.update(swc['radius'] / scaling)

    if pca_rot:
        print('Rotating x and y into their frame of maximal extent...')
        pca = PCA(copy=True)
        pc = np.vstack((swc['x'], swc['y'])).T
        pca.fit(pc)
        result = np.matmul(pc, pca.components_.T)

        swc.update(pd.DataFrame(result, columns=['x', 'y']))

    # create one point soma when there is more than three soma points
    sp = swc[swc['type'] == 1]
    if sp.shape[0] > 3:
        print('There are more than 3 soma points. The radius of the soma is estimated...')

        # calculate the convex hull of soma points
        convex_hull = ConvexHull(sp[['x', 'y', 'z']].values, qhull_options='QJ')

        hull_points = convex_hull.points

        centroid = np.mean(hull_points, axis=0)

        distances_to_centroid = np.linalg.norm(hull_points-centroid, axis=1)
        rad = np.max(distances_to_centroid)

        # fix the parent connections

        connection_locations = [row.name for k, row in swc.iterrows() if
                                row['parent'] in sp['n'].values and row['n'] not in sp['n'].values]
        connected_points = pd.concat([swc[swc['n'] == n] for n in connection_locations])
        connected_points['parent'] = 1
        swc.update(connected_points)

        to_delete = list(sp['n'].values)
        for k in connection_locations:
            try:
                to_delete.remove(k)
            except ValueError:
                continue

        soma_dict = dict(
            zip(['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], [int(1), int(1), centroid[0], centroid[1],
                                                                   centroid[2], rad, int(-1)]))

        swc.update(pd.DataFrame(soma_dict, index=[0]))

        # delete old soma points
        swc = swc.drop(swc.index[to_delete[:-1]])

        swc = swc.sort_index()

    # soma center on first entry
    centroid = swc[swc['n'] == 1][['x', 'y', 'z']].values.reshape(-1)
    swc.update(swc[['x', 'y', 'z']] - centroid)

    if soma_radius:
        print('Setting all nodes to type soma that have a larger radius than %s microns...' % soma_radius)

        d = np.vstack((swc['radius'], swc['type'])).T
        d[d[:, 0] >= soma_radius, 1] = 1
        swc.update(pd.DataFrame(d[:, 1].astype(int), columns=['type']))

    return swc


def smooth_gaussian(data, dim, sigma=2):
    """
        Smooths the given data using a gaussian. This method only works for stacked one or two dimensional data so
        far! Smoothing in 3D is not implemented.
        :param data: (X,Y,N) numpy.array 1,2 or 3 dimensional.
        :param dim: int
            Dimension of the passed data. Used to determine if data is a single image or stacked.
        :param sigma: int
            The standard deviation of the smoothing gaussian used.
        :return: Xb: same dimension as the input array.
            Smoothed data.
    """
    N=1
    if dim == 2:
        try:
            pX, pY, N = data.shape
        except ValueError:
            pX, pY = data.shape
            data = data.reshape(pX, pY, 1)

        # gaussian window
        win = gaussian(11, sigma)
        win = win * win.reshape(-1, 1)

        # add blur
        Xb = np.zeros((pX, pY, N))

        for k in range(N):
            Xb[:, :, k] = convolve2d(data[:, :, k], np.rot90(win), mode='same')

    elif dim == 1:

        try:
            pX, N = data.shape
        except ValueError:
            pX = data.shape[0]
            data = data.reshape(pX, 1)

        # add blur
        Xb = np.zeros((pX, N))

        for k in range(N):
            Xb[:, k] = gaussian_filter(data[:, k], sigma=sigma)
    else:
        raise NotImplementedError("There is no gaussian smoothing implemented for {0} dimensions".format(dim))

    if N == 1:
        Xb = np.squeeze(Xb)

    return Xb


def load_data(path, restriction=None, fl='density_map'):
    for root, _, files in os.walk(path):
        if fl in ['density_map', 'persistence']:
            data = []
            types = []
            file_names = []
            for f in files:
                t = f.split("_")[0]
                if (restriction and t in restriction) or not restriction:
                    a = np.loadtxt(root + '/' + f).reshape(-1)
                    data.append(list(a))
                    types.append(t)
                    file_names.append(f)
        elif fl == 'morphometrics':
            d = pd.read_csv(path + 'morphometrics.csv')

            # create index from restrictions
            idx = None
            for r in restriction:
                idx = np.logical_or(idx, d['type'] == r)
            # get data
            file_names = d[idx]['name']
            types = d[idx]['type']
            data = d[idx][['branch_points', 'depth', 'height', 'max_branch_angle', 'max_branch_order', 'max_path_angle',
                           'max_path_dist_to_soma', 'max_segment_length', 'max_tortuosity', 'mean_branch_angle',
                           'mean_path_angle', 'mean_tortuosity',
                           'stems', 'tips', 'total_length', 'width']].values

        elif fl == 'ephys':
            d = pd.read_csv(path + 'patch-morph-ephys-features.csv')
            file_names = d['name sample']
            types = d['Cell type']
            data = d.values[:, 1:-1].astype(float)
            #log transform AI/100 
            ind_ai = d.columns[1:-1] == 'AI (%)'
            data[:,ind_ai] = np.log(data[:,ind_ai]/100)

            # log transform Latency
            ind_latency = d.columns[1:-1] == 'Latency (ms)'
            data[:,ind_latency] = np.log(data[:,ind_latency])

        return pd.DataFrame(dict(file=file_names, type=types)), np.array(data)

def get_persistence_image(data, dim=2, t=1, bins=100, xmax=None, ymax=None):
    if not data.empty:
        if dim == 1:
            y = np.zeros((bins,))
            if xmax is None:
                xmax = np.max(data['birth'])
            x = np.linspace(0, xmax, bins)

            for k, p in data.iterrows():
                m = np.abs(p['birth'] - p['death'])
                y += m * norm.pdf(x, loc=p['birth'], scale=t)
            return y, x

        elif dim == 2:

            kernel = gaussian_kde(data[['birth', 'death']].values.T)

            if xmax is None:
                xmax = np.max(data['birth'].values)
            if ymax is None:
                ymax = np.max(data['death'].values)
            X, Y = np.mgrid[0:xmax:xmax / bins, 0:ymax:ymax / bins]
            X = X[:bins, :bins]
            Y = Y[:bins, :bins]

            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kernel(positions).T, X.shape)
            return Z, [np.unique(X), np.unique(Y)]


def get_std_dev(obs):
    std_dev = np.std(obs.astype(np.float64), axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
    return std_dev


def get_pca_transformed_data(X_trn, X_tst, pca, fm_lengths):

    indices = np.cumsum(fm_lengths)
    output_dims = np.array([k if k < 10 else 10 for k in fm_lengths])

    X_train_ = np.zeros((X_trn.shape[0], np.sum(output_dims)))
    X_test_ = np.zeros((X_tst.shape[0], np.sum(output_dims)))
    std_dev = np.ones((np.sum(output_dims)))

    for k in range(1, fm_lengths.shape[0]):
        start = indices[k - 1]
        stop = indices[k]
        o_start = np.cumsum(output_dims)[k - 1]
        o_stop = np.cumsum(output_dims)[k]
        if fm_lengths[k] > 10:  # always perform PCA when the length of the feature map exceeds 10

            X_train_[:, o_start:o_stop] = pca.fit_transform(X_trn[:, start:stop])
            X_test_[:, o_start:o_stop] = pca.transform(X_tst[:, start:stop])
        else:
            X_train_[:, o_start:o_stop] = X_trn[:, start:stop]
            X_test_[:, o_start:o_stop] = X_tst[:, start:stop]

        # always perform z-scoring
        std_dev[o_start:o_stop] = get_std_dev(X_train_[:, o_start:o_stop])

    X_train = copy.copy(X_train_) / std_dev
    X_test = copy.copy(X_test_) / std_dev

    if len(X_train.shape) > 2:
        X_train.squeeze(), X_test.squeeze()
    return X_train, X_test
