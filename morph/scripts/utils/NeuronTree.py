import copy
from os import makedirs
from os.path import exists
import sys
sys.setrecursionlimit(10000)

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random

from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy import stats

from shapely.geometry import MultiLineString, LineString, Point

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA
from .utils import angle_between


matplotlib.rcParams.update({'font.size': 14})


class NeuronTree:

    # resample nodes along the edges of G in distance d, return array of 3D positions
    @staticmethod
    def resample_nodes(G, d):
        P = []
        for (u, v, edata) in G.edges(data=True):
            n1 = G.node[u]
            n2 = G.node[v]
            e = edata['euclidean_dist']
            m = n2['pos'] - n1['pos']
            m /= np.linalg.norm(m)

            g = lambda x: x * m + n1['pos']

            for a in range(1, int(e / d)):
                P.append(g(a * d))

        P += list(nx.get_node_attributes(G, 'pos').values())
        return np.array(P)

    # creates a networkX Tree out of a swc file.
    # scaling denotes the conversion factor needed to convert the units given in swc file to microns
    # soma_rad denotes the radius of the soma given in microns
    def __init__(self, swc=None, scaling=1., soma_rad=1., node_data=[], edge_data=[], graph=None, post_process=True):
        # initialize tree DIRECTED
        if graph:
            G = graph
        else:
            G = nx.DiGraph()

            if swc is not None:
                node_keys = ['pos', 'type', 'radius']

                if type(swc) == pd.DataFrame:

                    # sort out node data
                    n = swc['n'].values
                    pos = np.array([swc['x'].values, swc['y'].values, swc['z'].values]).T / scaling
                    radius = swc['radius'].values / scaling
                    t = swc['type'].values
                    # if the radius is too big it belongs to the soma
                    if soma_rad:
                        t[swc['radius'] / scaling > soma_rad] = 1.
                    pid = swc['parent'].values

                elif type(swc) == np.ndarray:
                    n = swc['n']
                    pos = np.array([swc['x'], swc['y'], swc['z']]).T / scaling
                    if soma_rad:
                        swc['type'][swc['radius'] / scaling > soma_rad] = 1.
                    t = swc['type']
                    radius = swc['radius'] / scaling
                    pid = swc['parent']

                else:
                    raise ValueError('type of swc representation unknown')

                # create node data
                t[pid == -1] = 1
                node_data = list(zip(n,
                                     [dict(zip(node_keys, [pos[ix], t[ix], radius[ix]])) for ix in range(pos.shape[0])]))

                # create edge data
                n_ = n.tolist()
                parent_idx = [n_.index(pid[ix]) for ix in range(1,len(pid))]
                ec = np.sqrt(np.sum((pos[parent_idx] - pos[1:]) ** 2, axis=1))
                edge_keys = ['euclidean_dist', 'path_length']

                edge_data = list(zip(pid[1:], n[1:],
                                     [dict(zip(edge_keys, [ec[ix], ec[ix]])) for ix in range(ec.shape[0])]))

            G.add_nodes_from(node_data)
            G.add_edges_from(edge_data)

        self._G = G

        if G.nodes():

            self._remove_redundant_nodes()
            self._make_tree()
            if 'type' in self.get_node_attributes() and post_process:
                self._merge_roots_by_type()
                self._unify_type()
                self._clean_axon()

    def _merge_roots_by_type(self):

        R = self._G
        nodes_to_merge = True
        root_ix = self.get_root()
        root = R.node[root_ix]

        while nodes_to_merge:
            nodes_to_merge = False
            for succ in R.successors(root_ix):
                s = R.node[int(succ)]

                if s['type'] == root['type']:
                    nodes_to_merge = True
                    for e in R.successors(succ):
                        n2 = R.node[int(e)]
                        d = np.sqrt(np.sum((root['pos'] - n2['pos']) ** 2))
                        R.add_edge(root_ix, e, euclidean_dist=d, path_length=d)
                    R.remove_node(succ)
        self._G = R

    def _remove_redundant_nodes(self):
        """
        remove nodes that are at the same position
        :return: None
        """
        nodeindices = np.array(list(nx.get_edge_attributes(self._G, 'euclidean_dist').values())) == 0
        edgelist = list(np.array(list(nx.get_edge_attributes(self._G, 'euclidean_dist').keys()))[nodeindices])
        while edgelist:
            predecessor, redundantNode = edgelist.pop()

            n1 = self._G.node[predecessor]
            successors = self._G.successors(redundantNode)

            for succ in successors:
                n2 = self._G.node[succ]
                d = np.sqrt(np.sum((n1['pos'] - n2['pos']) ** 2))
                self._G.add_edge(predecessor, succ, euclidean_dist=d, path_length=d)

            # remove node from graph
            self._G.remove_node(redundantNode)
            nodeindices = np.array(list(nx.get_edge_attributes(self._G, 'euclidean_dist').values())) == 0
            edgelist = list(np.array(list(nx.get_edge_attributes(self._G, 'euclidean_dist').keys()))[nodeindices])

    def _merge_roots_by_distance(self, dist):

        R = self._G
        nodes_to_remove = True
        root_ix = self.get_root()
        root = R.node[root_ix]

        # merge everything that is dist microns away from root node pos
        # dist for BC cells is < 50 voxel = 2.5 microns
        while (nodes_to_remove):
            nodes_to_remove = False
            for succ in R.successors(root_ix):
                edge = R.get_edge_data(root_ix, succ)
                if edge['euclidean_dist'] <= dist:
                    nodes_to_remove = True
                    for e in R.successors(succ):
                        n2 = R.node[int(e)]
                        d = np.sqrt(np.sum((root['pos'] - n2['pos']) ** 2))
                        R.add_edge(root_ix, e, euclidean_dist=d, path_length=d)
                    R.remove_node(succ)

        self._G = R

    # merges split single paths into one only based on the out degree of the nodes. E.g.: A-->B-->C-->D becomes A-->D
    def _merge_edges_on_path_by_degree(self):

        Tree = self._G

        O = Tree.out_degree()
        d = [i == 1 for i in list(O.values())]
        I = np.array(list(O.keys()))
        if d:
            L = list(I[np.array(d)])  # contains all node ids with out degree 1
            try:
                L.remove(1)
            except ValueError:
                pass
        else:
            raise ValueError('d is empty, check the swc file!')
        D = copy.copy(L)

        while L:

            c = L.pop()
            p = Tree.predecessors(c)[0]
            s = Tree.successors(c)[0]

            path = Tree[c][s]['path_length'] + Tree[p][c]['path_length']
            while p in L:
                L.remove(p)
                c = p
                p = Tree.predecessors(c)[0]
                path += Tree[p][c]['path_length']

            # create edge
            pre = Tree.node[p]
            suc = Tree.node[s]
            d = np.sqrt(np.sum((pre['pos'] - suc['pos']) ** 2))
            Tree.add_edge(p, s, euclidean_dist=d, path_length=path)

        Tree.remove_nodes_from(D)
        self._G = Tree

    def _merge_edges_on_path_by_degree_recursive(self, start=1):

        def recursive_merging(Tree, current_node, start):
            deg = Tree.out_degree()[current_node]
            if deg == 1:
                succ = Tree.successors(current_node)[0]

                recursive_merging(Tree, succ, start)
                if start != current_node:
                    Tree.remove_node(current_node)
            else:
                if start != current_node and (start, current_node) not in Tree.edges():
                    path = nx.shortest_path_length(Tree, start, current_node, weight='path_length')
                    n1 = Tree.node[start]
                    n2 = Tree.node[current_node]
                    d = np.sqrt(np.sum((n1['pos'] - n2['pos']) ** 2))
                    Tree.add_edge(start, current_node, euclidean_dist=d, path_length=path)

                for succ in Tree.successors(current_node):
                    recursive_merging(Tree, succ, current_node)

        recursive_merging(self._G, start, start)

    # merges split single paths into one based on the out degree of the nodes and the distance
    # of the next node (C) to the projected line of the previous and the current node (AB)
    # E.g.: A-->B-->C becomes A-->C if C is within distance 'dist' of the line connecting A and B
    def _merge_edges_on_path_by_displacement(self, start=1, disp=5):

        Tree = self._G
        current_node = start
        successors = Tree.successors(start)

        while (successors):

            pred = Tree.predecessors(current_node)
            succ = successors.pop(0)
            deg = Tree.out_degree()[current_node]
            if deg == 1:
                if pred and pred == Tree.predecessors(Tree.predecessors(succ)[0]):
                    p = Tree.node[pred[0]]
                    s = Tree.node[succ]
                    c = Tree.node[current_node]
                    d = np.linalg.norm(np.cross(c['pos'] - p['pos'], p['pos'] - s['pos'])) / np.linalg.norm(
                        c['pos'] - p['pos'])
                    if d < disp:
                        for s in Tree.successors(current_node):
                            path = nx.shortest_path_length(Tree, pred[0], s, weight='path_length')
                            d = np.sqrt(np.sum((p['pos'] - s['pos']) ** 2))
                            Tree.add_edge(pred[0], s, euclidean_dist=d, path_length=path)

                        Tree.remove_node(current_node)
            S = Tree.successors(succ)
            S[len(S):] = successors
            successors = S

            current_node = succ
        self._G = Tree

    def _merge_edges_on_path_by_edge_length(self, start=1, e=0.01):

        Tree = self._G
        current_node = start
        successors = list(Tree.successors(start))

        while (successors):

            pred = list(Tree.predecessors(current_node))
            succ = successors.pop(0)
            deg = Tree.out_degree()[current_node]

            if deg == 1:
                if pred and pred == Tree.predecessors(list(Tree.predecessors(succ))[0]):
                    p = Tree.node[pred[0]]
                    s = Tree.node[succ]

                    d = np.sqrt(np.sum((s['pos'] - p['pos']) ** 2))

                    d1 = Tree.edge[pred[0]][current_node]['euclidean_dist']
                    d2 = Tree.edge[current_node][succ]['euclidean_dist']
                    dist = d1 + d2

                    if dist - d < e:
                        for s in Tree.successors(current_node):
                            path = nx.shortest_path_length(Tree, pred[0], s, weight='path_length')
                            Tree.add_edge(pred[0], s, euclidean_dist=d, path_length=path)

                        Tree.remove_node(current_node)
            S = list(Tree.successors(succ))
            S[len(S):] = successors
            successors = S

            current_node = succ
        self._G = Tree

    def _get_branch_type(self, B):
        Y = nx.get_node_attributes(self._G, 'type')
        bt = []
        bt += [Y[k] for k in B]
        return np.round(np.mean(bt))

    def _unify_type(self):

        for s in self._G.successors(1):
            S = nx.dfs_tree(self._G, s).nodes()
            t = self._get_branch_type(S)
            for k in S:
                self._G.node[k]['type'] = t

    def _clean_axon(self):
        # clean up axon: only longest axon is kept
        axon_edges = self.get_axon_edges()

        if axon_edges:
            axon_edges = np.array(axon_edges)
            edges = axon_edges[(axon_edges[:, 0] == 1)]
            l = []

            for n in edges:
                l.append(len(nx.dfs_tree(self._G, n[1]).nodes()))

            m = max(l)
            ax = edges[l.index(m)][1]
            to_remove = edges[(edges[:, 1] != ax)]

            for e in to_remove:
                self._G.remove_nodes_from(nx.dfs_tree(self._G, e[1]).nodes())

    def _clean_dendrites(self):

        # clean up dendrites: only keep dendrites that are longer than one node
        den_n = [k for k in self._G.successors(1) if self._G.node[int(k)]['type'] == 3]

        for n in den_n:
            if not (self._G.successors(n)):
                self._G.remove_node(n)

    def make_pos_relative(self):
        root = self.get_root()
        if 'pos' in self.get_node_attributes():
            root_pos = self._G.node[root]['pos']

            for v in self._G.node:
                v_pos = self._G.node[v]['pos']
                self._G.node[v]['pos'] = v_pos - root_pos
        else:
            raise Warning('There is no position data assigned to the nodes.')

    def _make_tree(self):

        G = self._G
        r = self.get_root()
        T = nx.dfs_tree(G, r)

        for n_attr in self.get_node_attributes():
            attr = nx.get_node_attributes(G, n_attr)
            nx.set_node_attributes(T, n_attr, attr)

        for e_attr in self.get_edge_attributes():
            attr = nx.get_edge_attributes(G, e_attr)
            nx.set_edge_attributes(T, e_attr, attr)

        self._G = T

    def get_graph(self):
        return self._G

    def get_mst(self):
        """
        Returns the minimal spanning tree of the Neuron.
        :return:
            NeuronTree: mst
        """
        # get the included nodes
        other_points = np.unique(np.append(self.get_branchpoints(), self.get_root()))
        tips = self.get_tips()

        # get the node data
        node_data = self.get_graph().node
        node_data_new = [(node, node_data[node]) for node in np.append(other_points, tips)]

        # get parent of each node and create edge_data
        nodes = set(tips)
        edge_data = self.get_graph().edge
        edge_data_new = []
        while nodes:
            current_node = nodes.pop()
            # node is soma
            if current_node != self.get_root():
                cn = copy.copy(current_node)
                pred = nx.DiGraph.predecessors(self.get_graph(), current_node)[0]
                path_length = edge_data[pred][cn]['path_length']
                while pred not in other_points:
                    cn = pred
                    pred = nx.DiGraph.predecessors(self.get_graph(), pred)[0]
                    path_length += edge_data[pred][cn]['path_length']

                ec = np.sqrt(np.sum((node_data[current_node]['pos'] - node_data[pred]['pos']) ** 2))
                # add edge to edge_data_new
                edge_data_new.append((pred, current_node, dict(euclidean_dist=ec, path_length=path_length)))

                nodes.add(pred)  # adds the predecessor only once since nodes is a set

        return NeuronTree(node_data=node_data_new, edge_data=edge_data_new)

    def get_node_attributes(self):
        """ returns the list of attributes assigned to each node.
            If no attributes are assigned it returns an empty list.

            :return:
                list : attribute names assigned to nodes
        """
        attr = []
        if self._G.nodes():
            node_id = self._G.nodes()[0]
            attr = list(self._G.node[node_id].keys())
        return attr

    def get_edge_attributes(self):
        """
            Returns the list of attributes assigned to each edge.
            If no attributes are assigned it returns an empty list.
        :return:
            list : attribute names assigned to edges
        """
        attr = []
        if self._G.edges():
            e = self._G.edges()[0]
            attr = list(self._G.edge[e[0]][e[1]].keys())
        return attr

    def get_root(self):
        try:
            root = np.min(self.nodes(type_ix=1))
        except (ValueError, KeyError):
            print('No node is attributed as being the soma. Returning the smallest node id.')
            root = np.min(self.nodes())

        return root

    def reduce(self, method='mst', e=0.01):
        """
        Reduces the number of nodes in the given tree by pruning nodes on paths like A-->B-->C. B is pruned and the
        resulting edge A-->C is inserted under circumstances defined by the keyword 'method'.
        :param method: (default 'mst') Defines the method by which the number of nodes is reduced. Possible methods are
                'mst' -- deletes all nodes in between branch points. Results in the minimal spanning tree of the neuron
                representation.
                'dist' -- deletes all nodes B on path A-->B-->C that do not change the length of edge A-->C by the
                amount of epsilon e
                (in microns).
                'disp' -- deletes all nodes B on path A-->B-->C that are maximally e microns displaced from the edge
                A-->C.
        :param e: float (default 0.01) error margin for methods 'dist' and 'disp'. e is interpreted in microns.
        :return: None. The tree is pruned in place. If the original tree is desired to be conserved then copy the tree
        data beforehand via the copy/deepcopy constructor.
        """

        if type(self._G) == nx.classes.graph.Graph:
            self._make_tree()

        if method == 'mst':
            self._merge_edges_on_path_by_degree()
        elif method == 'dist':
            self._merge_edges_on_path_by_edge_length(e=e)
        elif method == 'disp':
            self._merge_edges_on_path_by_displacement(disp=e)
        else:
            raise NotImplementedError('Method {0} is not implemented!'.format(method))

    def truncate_nodes(self, perc=.1, no_trunc_nodes=None):
        """
        Truncates the number of nodes by the given percentage. The nodes are pruned equally from the tips inward until
        the approximated percentage has been deleted or at least the given number of nodes is truncated.
        :param perc: float. Percent of nodes that are to be truncated between 0 and 1. Default=0.1
        :param no_trunc_nodes: int . Optional. Number of nodes to be truncated
        :return: The truncated tree.
        """
        T = copy.copy(self)
        nodes = self.nodes()
        total_no_nodes = len(nodes)

        if not no_trunc_nodes:
            no_trunc_nodes = int(total_no_nodes * perc)

        tips = T.get_tips()

        for t in tips:
            nodes.remove(t)

        while total_no_nodes - no_trunc_nodes < len(nodes):
            T = NeuronTree(graph=T.get_graph().subgraph(nodes))
            nodes = T.nodes()

            tips = T.get_tips()

            for t in tips:
                nodes.remove(t)
        
        return T

    def nodes(self, type_ix=None, data=False):

        if type_ix is None:
            nodes = self._G.nodes(data=data)
        else:
            nodes = [k for k in self._G.node if self._G.node[k]['type'] == type_ix]
        return nodes

    def edges(self, start=None, type_ix=None, data=False):
        if type_ix is None:
            edges = self._G.edges(start, data=data)
        else:
            nodes = self.nodes(type_ix=type_ix)
            edges = [x for x in self._G.edges(start, data=data) if (x[0] in nodes or x[1] in nodes)]
        return edges

    def get_dendrite_nodes(self, data=False):
        return np.array(self.nodes(type_ix=3,data=data))

    def get_axon_nodes(self, data=False):
        return np.array(self.nodes(type_ix=2, data=data))

    def get_axon_edges(self, start=None, data=False):
        axon_nodes = self.get_axon_nodes()
        return [x for x in self._G.edges(start,data=data) if (x[0] in axon_nodes or x[1] in axon_nodes)]

    def get_dendrite_edges(self, start=None, data=False):
        dendrite_nodes = self.get_dendrite_nodes()
        return [x for x in self._G.edges(start,data=data) if (x[0] in dendrite_nodes or x[1] in dendrite_nodes)]

    def get_tips(self):
        E = self._G.edge
        return np.array([e for e in E if E[e] == {}])

    def get_branchpoints(self):
        bp_indx = np.where(np.array(np.sum(nx.adjacency_matrix(self._G), axis=1)).flatten() > 1)
        return np.array(self.nodes())[bp_indx]

    def get_dendritic_tree(self):
        nodes = list(self.get_dendrite_nodes())
        nodes.insert(0, self.get_root())
        subgraph = nx.subgraph(self._G, nodes)
        return NeuronTree(graph=subgraph)

    def get_axonal_tree(self):
        nodes = list(self.get_axon_nodes())
        nodes.insert(0, self.get_root())
        subgraph = nx.subgraph(self._G, nodes)
        return NeuronTree(graph=subgraph)

    # returns the adjacency matrix of the Tree saved in self._G. weight can be None, 'euclidean_dist' or 'path_length'
    def get_adjacency_matrix(self, weight=None):
        return nx.adjancency_matrix(self._G, weight=weight)

    def get_extend(self):
        P = np.array(list(nx.get_node_attributes(self._G, 'pos').values()))
        return np.max(P, axis=0) - np.min(P, axis=0)

    def get_root_angle_dist(self, angle_type='axis_angle', **kwargs):
        """
               Returns the histogram over the root angle distribution over a tree. Root angle denotes the orientation of
               each edge with respect to the root.
               :param self: NeuronTree object
               :param bins: int
                    Number of bins used in the histogram. Default is 10.
               :param angle_type: either 'axis_angle' or 'euler'
                    Defines the type of angle that is calculated. Euler angles are defined as angles around the canonical
                    euler axes (x, y and z). Axis angles are defined with respect to the rotation axis between two
                    vectors.
               :returns:
                    hist: histogram over angles
                    edges: edges of histogram. For further information see numpy.histogramdd()
           """
        from utils.utils import angle_between, get_rotation_matrix, rotationMatrixToEulerAngles

        angles = []
        if angle_type == 'axis_angle':
            dim = 1
            func = lambda u,v: angle_between(u, v)
        elif angle_type == 'euler':
            dim =3
            func = lambda u,v: rotationMatrixToEulerAngles(get_rotation_matrix(u, v))
        else:
            raise NotImplementedError('Angle type %s is not implemented' % angle_type)

        for n1, n2 in self._G.edges():
            u = self._G.node[n2]['pos'] - self._G.node[n1]['pos']
            v = self._G.node[n1]['pos'] - self._G.node[self.get_root()]['pos']
            angles.append(func(u, v))
        angles = np.array(angles)
        hist = np.histogramdd(angles, range=[[0, np.pi]]*dim, **kwargs)
        return hist

    def get_branch_order(self):
        """
        Returns the dictionary of the branch order of each node.
        :return:
            d: dict
            Dictionary of the form {u: branch_order} for each node.
        """
        return self._get_branch_order(1,0)

    def _get_branch_order(self, start, bo):
        """
        Returns the dictionary assigning the right branch order to each node.
        :param start: starting node
        :param bo: starting branch order
        :return:
            d: dict
            Dictionary of the form {u: branch_order} for each node reachable from starting node.
        """
        d = {}
        edges = self.edges(start)
        d[start] = bo
        if len(edges) > 1:
            for e in edges:
                d.update(self._get_branch_order(e[1], bo + 1))
        elif len(edges) == 1:
            d.update(self._get_branch_order(edges[0][1], bo))
        return d

    def _get_distance(self, dist, as_dict=False):
        if dist == 'path_from_soma':

            if as_dict:
                dist_ = nx.single_source_dijkstra_path_length(self.get_graph(), source=self.get_root(),
                                                              weight='euclidean_dist')
            else:
                dist_ = np.array(list(nx.single_source_dijkstra_path_length(self.get_graph(), source=self.get_root(),
                                                                            weight='euclidean_dist').values()))
        elif dist == 'branch_order':
            dist_ = self._get_branch_order(1, 0)
            if not as_dict:
                dist_ = np.array(list(dist_.values()))

        else:
            raise NotImplementedError

        return dist_

    def get_segment_length(self):
        """
        Returns the dictionary of the segment length in microns of each branch. The keys of the dictionary denote the tuples of the
        starting and end node of each segment.
        :return:
            d: dict
            Dictionary of the form {(n_s, n_e): segment length[u] }
        """

        G = copy.copy(self.get_graph())
        T = NeuronTree(graph=G)
        T.reduce()

        dist = lambda x, y: np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

        segment_length = {e: dist(G.node[e[0]]['pos'], G.node[e[1]]['pos']) for e in T.edges()}
        return segment_length

    def get_kde_distribution(self, key, dist=None):
        """

        :param key:
        :param dist:
        :return:
        """

        if key == 'thickness':
            thickness = np.array(list(nx.get_node_attributes(self.get_graph(), 'radius').values()))

            if dist:
                data = np.array(list(zip(self._get_distance(dist), thickness)))
                return stats.gaussian_kde(data)

            else:
                return stats.gaussian_kde(thickness)
        elif key == 'path_angle':

            successors = nx.dfs_successors(self.get_graph(), 1)
            path_angle = []
            nodes = []
            for n1, n2 in self.edges():
                u = self.get_graph().node[n2]['pos'] - self.get_graph().node[n1]['pos']
                try:
                    for succ in successors[n2]:
                        v = self.get_graph().node[succ]['pos'] - self.get_graph().node[n2]['pos']
                        path_angle.append(angle_between(u, v) * 180 / np.pi)  # convert angles into degree
                        nodes.append(n2)
                except KeyError:
                    continue

            if dist:
                distance = self._get_distance(dist, as_dict=True)
                distance = [distance[n] for n in nodes]

                data = np.array(list(zip(distance, path_angle)))
                return stats.gaussian_kde(data)
            else:
                return stats.gaussian_kde(path_angle)

        elif key == 'branch_angle':
            branch_angles = []
            branchpoints = self.get_branchpoints()
            for bp in branchpoints:
                successors = list(self.get_graph().edge[bp].keys())
                branches = []
                for succ in successors:  # create a vector for each branching edge
                    v = self.get_graph().node[succ]['pos'] - self.get_graph().node[bp]['pos']
                    branches.append(v)
                for u, v in combinations(branches, 2):
                    branch_angles.append(angle_between(u, v) * 180 / np.pi)

            if dist:
                distance = self._get_distance(dist, as_dict=True)
                distance = [distance[bp] for bp in branchpoints]
                data = np.array(list(zip(distance, branch_angles)))
                return stats.gaussian_kde(data)
            else:
                return stats.gaussian_kde(branch_angles)

        elif key == 'branch_order':
            branch_order = list(self._get_branch_order(1, 0).values())

            if dist:
                data = np.array(list(zip(self._get_distance(dist), branch_order)))
                return stats.gaussian_kde(data)
            else:
                return stats.gaussian_kde(branch_order)

        elif key == 'distance':
            distance = np.array(list(nx.single_source_dijkstra_path_length(self.get_graph(), source=self.get_root(),
                                                                           weight='euclidean_dist').values()))

            if dist:
                data = np.array(list(zip(self._get_distance(dist), distance)))
                return stats.gaussian_kde(data)
            else:
                return stats.gaussian_kde(distance)
        else:
            raise NotImplementedError

    def get_distance_dist(self, **kwargs):
        """
            Returns histogram of the distance distribution from soma
        :param kwargs: optional parameters passed to histogram calculation (see numpy.histogramdd)
        :return:
            hist    1D histogram
            edges   bin edges of the histogram in hist. Definition as in numpy.histogrammdd
        """

        dist = np.array(list(nx.single_source_dijkstra_path_length(self.get_graph(), source=self.get_root(),
                                                                   weight='euclidean_dist').values()))

        return np.histogram(dist, **kwargs)

    def get_branch_order_dist(self, **kwargs):
        """
            Returns histogram of the branch order distribution
        :param kwargs: optional parameters passed to histogram calculation (see numpy.histogramdd)
        :return:
            hist    1D histogram
            edges   bin edges of the histogram in hist. Definition as in numpy.histogrammdd
        """
        data = list(self._get_branch_order(1, 0).values())

        # TODO make it assignable a dist_measure
        return np.histogram(data, **kwargs)

    def get_thickness_dist(self, dist_measure=None, **kwargs):
        """
            Returns the distribution of the neurons thickness in microns in form of a histogram.
            The distribution can be calculated against a distance measure, namely 'path_from_soma' or 'branch_order'.
        :param dist_measure:
        :param kwargs:
        :return:
            hist    1D or 2D histogram
            edges   bin edges used
        """
        thickness = np.array(list(nx.get_node_attributes(self.get_graph(), 'radius').values()))

        if dist_measure:

            dist = self._get_distance(dist_measure)
            data = np.array(list(zip(dist, thickness)))
            return np.histogramdd(data, **kwargs)
        else:

            return np.histogram(thickness, **kwargs)

    def get_path_angles(self):
        """
        Returns a dictionary of path angles between two edges. Angles are reported in degree
        :return: dict of path angles btw two edges.
                d[u][v][w] returns the angle between edge (u,v) and (v,w)
        """
        successors = nx.dfs_successors(self.get_graph(), 1)
        path_angle = {}
        for n1, n2 in self.edges():
            u = self.get_graph().node[n2]['pos'] - self.get_graph().node[n1]['pos']
            path_angle[n1] = {}
            try:
                path_angle[n1][n2] = {}
                for succ in successors[n2]:
                    v = self.get_graph().node[succ]['pos'] - self.get_graph().node[n2]['pos']
                    path_angle[n1][n2][succ] = angle_between(u, v) * 180 / np.pi
            except KeyError:
                continue

        return path_angle

    def get_path_angle_dist(self, dist_measure=None, **kwargs):
        """
            Returns the distribution of the path angle, so the angles that are made between two consecutive segments.
            The distribution can be calculated against a distance measure, namely 'path_from_soma' or 'branch_order'.
            The path angles are returned in degree.
        :param dist_measure: string. default = None
            Defines the distance measure against whom the thickness is calculated. Possible choices are 'path_from_soma'
            or 'branch_order'.
        :param kwargs: additional arguments to be passed to histogram calculation
        :return:
            hist    1- or 2D histogram
            edges   bin edges of the histogram in hist. Definition as in numpy.histogrammdd
        """

        successors = nx.dfs_successors(self.get_graph(), 1)
        path_angle = []
        nodes = []
        for n1, n2 in self.edges():
            u = self.get_graph().node[n2]['pos'] - self.get_graph().node[n1]['pos']
            try:
                for succ in successors[n2]:
                    v = self.get_graph().node[succ]['pos'] - self.get_graph().node[n2]['pos']
                    path_angle.append(angle_between(u, v) * 180 / np.pi)  # convert angles into degree
                    nodes.append(n2)
            except KeyError:
                continue

        if dist_measure:
            distances = self._get_distance(dist_measure, as_dict=True)
            dist = [distances[n] for n in nodes]
            data = np.array(list(zip(dist, path_angle)))
            return np.histogramdd(data, **kwargs)
        else:
            return np.histogram(path_angle, **kwargs)

    def get_branch_angles(self):
        """
        Returns the list of branch angles.
        :return:
            branch_angles   list of branch angles.
        """

        from itertools import combinations

        branch_angles = []
        branchpoints = self.get_branchpoints()
        for bp in branchpoints:
            successors = list(self.get_graph().edge[bp].keys())
            branches = []
            for succ in successors:  # create a vector for each branching edge
                v = self.get_graph().node[succ]['pos'] - self.get_graph().node[bp]['pos']
                branches.append(v)
            for u, v in combinations(branches, 2):
                branch_angles.append(angle_between(u, v) * 180 / np.pi)
        return branch_angles

    def get_branch_angle_dist(self, dist_measure=None, **kwargs):
        """
            Returns the distribution of the branch angles, so the angles that are made between two branching segments.
            The distribution is calculated against a distance measure, namely 'path_from_soma' or 'branch_order'.
            The branch angles are returned in degree.
        :param dist_measure: string. default = 'path_from_soma'
            Defines the distance measure against whom the thickness is calculated. Possible choices are 'path_from_soma'
            or 'branch_order'.
        :param kwargs: dditional arguments to be passed to histogram calculation
        :return:
            hist    2D histogram
            edges   bin edges of the histogram in hist. Definition as in numpy.histogrammdd
        """

        branch_angles = self.get_branch_angles()
        branchpoints = self.get_branchpoints()
        if dist_measure:
            distances = self._get_distance(dist_measure, as_dict=True)
            dist = [distances[n] for n in branchpoints]
            data = np.array(list(zip(dist, branch_angles)))
            return np.histogramdd(data, **kwargs)
        else:
            return np.histogramdd(branch_angles, **kwargs)

    def get_segment_length_dist(self, **kwargs):

        segment_length = list(self.get_segment_length().values())

        return np.histogram(segment_length, **kwargs)

    def get_volume(self):
        """
        Returns the volume for each segment in the tree.
        :return: dictionary of the form d[(u,v)] = volume(u,v)
        """

        d = {}
        for e in self.edges(data=True):

            h = e[2]['euclidean_dist']
            r = self.get_graph().node[e[0]]['radius']
            R = self.get_graph().node[e[1]]['radius']

            d[(e[0], e[1])] = (1/3)*np.pi*h*(r*r + r*R + R*R)
        return d

    def get_surface(self):
        """
        Returns the surface for each segment in the tree treating each edge as a pipe (without closing lids!).
        :return: dictionary of the form d[(u,v)] = surface(u,v)
        """
        d = {}
        for e in self.edges(data=True):
            h = e[2]['euclidean_dist']
            r = self.get_graph().node[e[0]]['radius']
            R = self.get_graph().node[e[1]]['radius']

            d[(e[0], e[1])] = np.pi*(r + R)*np.sqrt((R-r)**2 + h**2)
        return d

    def get_sholl_intersection_profile(self, proj='xy', steps=36, centroid='centroid'):

        G = self.get_graph()
        coordinates = []

        if proj == 'xy':
            indx = [0, 1]
        elif proj == 'xz':
            indx = [0, 2]
        elif proj == 'yz':
            indx = [1, 2]
        else:
            raise ValueError("Projection %s not implemented" % proj)

        # get the coordinates of the points
        for e in G.edges():
            p1 = np.round(G.node[e[0]]['pos'], 2)
            p2 = np.round(G.node[e[1]]['pos'], 2)
            coordinates.append((p1[indx], p2[indx]))

        # remove illegal points
        coords = [c for c in coordinates if (c[0][0] != c[1][0] and c[0][1] != c[1][1])]

        lines = MultiLineString(coords).buffer(0.001)
        bounds = np.array(lines.bounds).reshape(2, 2).T
        if centroid == 'centroid':
            center = np.array(lines.convex_hull.centroid.coords[0])
            p_circle = Point(center)
        elif centroid == 'soma':
            center = np.array((0, 0))
            p_circle = Point(center)
        else:
            raise ValueError("Centroid %s is not defined" % centroid)

        r_max = max(np.linalg.norm(center - bounds[:, 0]), np.linalg.norm(center - bounds[:, 1]))

        intersections = []
        intervals = [0]
        for k in range(1, steps + 1):

            r = (r_max / steps) * k
            c = p_circle.buffer(r).boundary

            i = c.intersection(lines)
            if type(i) in [Point, LineString]:
                intersections.append(1)
            else:
                intersections.append(len(i))
            intervals.append(r)

        return intersections, intervals

    def get_persistence(self):
        """
        Creates the persistence barcode for the graph G. The algorithm is taken from
        _Quantifying topological invariants of neuronal morphologies_ from Lida Kanari et al
        (https://arxiv.org/abs/1603.08432).

        :return: pandas.DataFrame with entries node_id | birth | death . Where birth and death are defined in radial
            distance from soma.
        """

        # Initialization
        L = self.get_tips()
        R = self.get_root()
        G = self.get_graph()
        D = dict(node_id=[], type=[], birth=[], death=[])  # holds persistence barcode
        v = dict()  # holds 'aging' function of visited nodes defined by f

        # active nodes
        A = list(copy.copy(L))

        # radial distance function
        root_pos = G.node[R]['pos']
        f = lambda n: np.sqrt(np.dot(G.node[n]['pos'] - root_pos, G.node[n]['pos'] - root_pos))

        # set the initial value for leaf nodes
        for l in L:
            v[l] = f(l)

        while R not in A:
            for l in A:
                p = G.predecessors(l)[0]
                C = G.successors(p)

                # if all children are active
                if all(c in A for c in C):
                    # choose randomly from the oldest children
                    age = np.array([v[c] for c in C])
                    indices = np.where(age == age[np.argmax(age)])[0]
                    c_m = C[random.choice(indices)]

                    A.append(p)

                    for c_i in C:
                        A.remove(c_i)
                        if c_i != c_m:
                            D['node_id'].append(c_i)
                            D['type'].append(G.node[c_i]['type'])
                            D['birth'].append(v[c_i])
                            D['death'].append(f(p))
                    v[p] = v[c_m]
        D['node_id'].append(R)
        D['type'].append(1)
        D['birth'].append(v[R])
        D['death'].append(f(R))
        return pd.DataFrame(D)

    def get_gillette_sequence(self, order='StL'):
        MST = self.get_mst()
        # first step: assign sequence type to each node and determine sub tree depth
        sequence_types = dict()
        subtree_length = dict()
        G = MST.get_graph()
        tips = MST.get_tips()

        for n in MST.nodes():
            S = nx.DiGraph.successors(G,n)

            b = np.array([s in tips for s in S])
            if b.all():
                t='T'
            elif b.any():
                t='C'
            else:
                t='A'

            sequence_types[n] = t
            subtree_length[n] = len(nx.dfs_tree(G, n).nodes())

        nx.set_node_attributes(G,name='s_type', values=sequence_types)
        nx.set_node_attributes(G,name='subtree_length', values=subtree_length)

        # traverse nodes in right order. Here: short then long
        no_nodes = len(G.nodes())
        S= [1]
        k = -1
        while len(S) != no_nodes:
            s= S[k]
            succ = [s_ for s_ in G.successors(s) if s_ not in S] # remove nodes that you have visited already    

            if len(succ) == 0:
                k -= 1
            else:

                idx = np.argsort([subtree_length[s_] for s_ in succ])
                if order == 'LtS':
                    #reverse since the longest get traversed first
                    idx = idx[::-1]
                S.append(np.array(succ)[idx][0])
                k = -1
        seq = ''.join([sequence_types[s] for s in S])
        return seq

    # re-sample new nodes along the tree in equidistant distance dist (given in microns)
    # all original branching points are kept!
    # returns the tuple Pos, PID, TYPE, RADIUS, DIST
    def _resample_tree_data(self, dist=1):
        P = []
        PID = []
        TYPE = []
        RADIUS = []
        DIST = []

        pid = -1

        for (u, v, edata) in self._G.edges(data=True):
            n1 = self._G.node[u]
            n2 = self._G.node[v]
            e = edata['euclidean_dist']
            v = n2['pos'] - n1['pos']
            v /= np.linalg.norm(v)

            # line between two nodes
            g = lambda x: x * v + n1['pos']

            # radius function

            if n1['radius'] == n2['radius']:
                r = lambda x: n2['radius']
            else:
                x = [0, e]
                y = [n1['radius'], n2['radius']]
                r = interp1d(x, y)

            n = list(n1['pos'])

            if n in P:
                pid = P.index(n)
            else:
                P.append(n)
                PID.append(pid)
                pid += 1
                TYPE.append(n1['type'])
                DIST.append(0)
                RADIUS.append(n1['radius'])

            for a in range(1, int(e / dist)):
                P.append(list(g(a * dist)))
                PID.append(pid)
                pid = len(P) - 1
                TYPE.append(n2['type'])
                DIST.append(dist)
                RADIUS.append(np.array([r(a * dist)])[0])

            P.append(list(n2['pos']))
            PID.append(pid)
            pid += 1
            TYPE.append(n2['type'])
            DIST.append(e % dist)
            RADIUS.append(n2['radius'])

        P = np.array(P)

        return P, PID, TYPE, RADIUS, DIST

    def resample_tree(self, dist=1):

        (pos, pid, t, r, d) = self._resample_tree_data(dist)
        n_attr = [dict(pos=pos[i], type=t[i], radius=t[i]) for i in range(len(d))]

        nodes = list(zip(range(1, len(pid) + 1), n_attr))

        e_attr = [dict(euclidean_dist=d[i], path_length=d[i]) for i in range(len(d))]
        edges = list(zip(np.array(pid) + 1, range(1, len(pid) + 1), e_attr))

        T = NeuronTree(node_data=nodes, edge_data=edges[1:])

        return T

    def get_histogramdd(self, decomposition='ica', dim=3, proj_axes=None, whiten=True,
                        nbins=100, r=None, sampling_dist=0.01):
        p = NeuronTree.resample_nodes(self._G, sampling_dist)

        if decomposition:
            if decomposition == 'pca':
                # find principal axes of the 3D point cloud using PCA
                pca = PCA(n_components=dim, whiten=whiten)
                results = pca.fit_transform(p)
            elif decomposition == 'ica':
                ica = FastICA(n_components=dim, whiten=whiten)
                results = ica.fit_transform(p)
            else:
                raise ValueError('decomposition {0} is not implemented.'.format(decomposition))
        else:
            if proj_axes:
                results = p[:, proj_axes]
            else:
                results = p
        if r:
            return np.histogramdd(results, bins=(nbins,) * dim, range=r, normed=True)
        else:
            return np.histogramdd(results, bins=(nbins,) * dim, normed=True)


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
    ########################################## DRAWING FUNCTIONS #############################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

    def get_node_colors(self):
        axon_nodes = self.get_axon_nodes()
        dendrite_nodes = self.get_dendrite_nodes()

        colors = []
        for node in self._G.nodes():
            if node in axon_nodes:
                colors.append('g')
            elif node in dendrite_nodes:
                colors.append('y')
            else:
                colors.append('grey')
        return colors

    def draw_3D(self, fig=None, ix=None, reverse=True, r_axis='z', axon_color='grey', dendrite_color='darkgrey'):

        nodes = [k for k in nx.get_node_attributes(self._G, 'pos').values()]
        nodes = np.array(nodes)

        t = [axon_color if k == 2 else dendrite_color for k in nx.get_node_attributes(self._G, 'type').values()]

        # plot G
        if not fig:
            fig = plt.figure()
            ix = 111
        ax = fig.add_subplot(ix, projection='3d')
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=t, marker='.')
        plt.hold
        root_pos = self._G.node[self.get_root()]['pos']
        ax.scatter(root_pos[0], root_pos[1], root_pos[2], c='k', marker='^')

        colors = ['k', axon_color, dendrite_color]
        V = np.zeros((len(self._G.edges()), 2, 3))
        cs = []
        for k, e in enumerate(self._G.edges()):
            n1 = self._G.node[e[0]]
            n2 = self._G.node[e[1]]
            v = np.array([n1['pos'],n2['pos']])

            ax.plot3D(v[:, 0], v[:, 1], v[:, 2], c=colors[int(n2['type']) - 1])

        ax = plt.gca()
        if reverse:
            if 'z' in r_axis:
                ax.set_zlim(ax.get_zlim()[::-1])
            if 'x' in r_axis:
                ax.set_xlim(ax.get_xlim()[::-1])
            if 'y' in r_axis:
                ax.set_ylim(ax.get_ylim()[::-1])

        ax.set_xlabel('X [microns]')
        ax.set_ylabel('Y [microns]')
        ax.set_zlabel('Z [microns]')

    def draw_3D_volumetric(self, fig=None, ix=None, axon_color='grey', dendrite_color='darkgrey'):

        from .utils import get_rotation_matrix

        P = nx.get_node_attributes(self._G, 'pos')
        Rad = nx.get_node_attributes(self._G, 'radius')
        Type = nx.get_node_attributes(self._G, 'type')

        num = 5

        u = np.linspace(0, 2 * np.pi, num=num)
        v = np.linspace(0, np.pi, num=num)

        # parametric surface of a sphere
        fx = lambda r: r * np.outer(np.cos(u), np.sin(v))
        fy = lambda r: r * np.outer(np.sin(u), np.sin(v))
        fz = lambda r: r * np.outer(np.ones(np.size(u)), np.cos(v))

        if not fig:
            fig = plt.figure()
            ix = 111

        ax = fig.add_subplot(ix, projection='3d')

        unit_z = np.array([0, 0, 1])
        # plot the nodes as spheres
        for i in self.nodes():
            pos = P[i]
            r = Rad[i]
            if Type[i] == 2:
                c = axon_color
            elif Type[i] == 1:
                c = 'k'
            else:
                c = dendrite_color
            ax.plot_surface(fx(r) + pos[0], fy(r) + pos[1], fz(r) + pos[2], color=c)

        # plot segments as cone frustums
        for e in self._G.edges():
            if e[0] == 1:
                a = Rad[e[1]]
                b = Rad[e[1]]
            else:
                a = Rad[e[0]]
                b = Rad[e[1]]

            if Type[e[1]] == 2:
                c = axon_color
            else:
                c = dendrite_color

            h = self._G.edge[e[0]][e[1]]['euclidean_dist']
            # translation
            T = P[e[0]]
            # rotation
            R = get_rotation_matrix(unit_z, P[e[1]] - P[e[0]])

            k = np.linspace(0, h, num)
            t = np.linspace(0, 2 * np.pi, num)

            # parametric surface of a cone frustrum
            cx = lambda k, t: np.outer((a * (h - k) + b * k) / h, np.cos(t))
            cy = lambda k, t: np.outer((a * (h - k) + b * k) / h, np.sin(t))

            F = np.array([cx(k, t), cy(k, t), np.meshgrid(k, k)[1]]).T

            R_ = np.reshape(np.tile(R.T, (1, num)).T, [num, 3, 3])
            E = np.einsum('lij, lkj->lki', R_, F)
            ax.plot_surface(E[:, :, 0] + T[0], E[:, :, 1] + T[1], E[:, :, 2] + T[2], color=c)

        ax.set_xlabel('X [microns]')
        ax.set_ylabel('Y [microns]')
        ax.set_zlabel('Z [microns]')

    def draw_tree(self, edge_labels=False, **kwds):

        pos = nx.drawing.nx_agraph.graphviz_layout(self._G, prog='dot')

        colors = self.get_node_colors()

        nx.draw_networkx(self._G, pos, node_color=colors, **kwds)
        if edge_labels:
            # draw graph with weights on edges
            edge_labels = {(n1, n2): self._G[n1][n2]['path_length'] for (n1, n2) in self._G.edges()}
            nx.draw_networkx_edge_labels(self._G, pos, edge_labels=edge_labels, **kwds)



    def draw_2D(self, fig=None, projection='xz', axon_color='grey', dendrite_color='darkgrey', x_offset=0, y_offset=0,
                **kwargs):

        if not fig:
            fig = plt.figure()
        if projection == 'xy':
            indices = [0,1]
        elif projection == 'xz':
            indices = [0,2]
        elif projection == 'yz':
            indices = [1,2]
        else:
            raise ValueError('projection %s is not defined.'% projection)

        G = self.get_graph()
        V = np.zeros((len(G.edges()), 2, 3))
        colors = ['k', axon_color, dendrite_color]

        cs = []
        for k, e in enumerate(G.edges()):
            n1 = G.node[e[0]]
            n2 = G.node[e[1]]
            V[k, 0] = n1['pos']
            V[k, 1] = n2['pos']
            cs.append(colors[int(n2['type']) - 1])

        cs = np.array(cs)
        x = indices[0]
        y = indices[1]

        plt_idx = cs == axon_color
        if len(plt_idx.shape) > 1:
            plt_idx = plt_idx[:, 0]

        p = plt.plot(V[plt_idx, :, x].T + x_offset, V[plt_idx, :, y].T + y_offset, c=axon_color, **kwargs)

        plt_idx = cs == dendrite_color
        if len(plt_idx.shape) > 1:
            plt_idx = plt_idx[:, 0]
        p = plt.plot(V[plt_idx, :, x].T + x_offset, V[plt_idx, :, y].T + y_offset, c=dendrite_color, **kwargs)


    ############# SAVING FUNCTIONS #####################

    def to_swc(self):
        # create dataframe with graph data
        G = self._G
        ids = G.nodes()
        pos = np.round(np.array(list(nx.get_node_attributes(G, 'pos').values())), 2)
        r = np.array(list(nx.get_node_attributes(G, 'radius').values()))
        t = np.array(list(nx.get_node_attributes(G, 'type').values())).astype(int)
        pids = [list(l.keys())[0] for l in list(G.pred.values()) if list(l.keys())]
        pids.insert(0, -1)
        # write graph into swc file
        d = {'n': ids, 'type': t, 'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2], 'radius': r, 'parent': pids}
        df = pd.DataFrame(data=d, columns=d.keys())

        return df

    def write_to_swc(self, file_name,
                     ext='',
                     path='/gpfs01/berens/data/data/anatomy/BC_morphologies/swc_tree/'):

        if not exists(path + ext):
            makedirs(path + ext)

        df = self.to_swc()
        df.to_csv(path + ext + file_name + '.swc', sep=' ', encoding='utf-8', header=False, index=False)

    def write_to_mat(self, file_name,
                     ext='',
                     path='/gpfs01/berens/data/data/anatomy/BC_morphologies/csv_luxburg/'):

        if not exists(path + ext):
            makedirs(path + ext)

        data = {}
        G = self._G
        P = list(nx.get_node_attributes(G, 'pos').values())
        T = list(nx.get_node_attributes(G, 'type').values())
        A = nx.adjacency_matrix(G, weight='path_length')

        data['pos'] = P
        data['type'] = T
        data['A'] = A
        savemat(path + ext + file_name, data)