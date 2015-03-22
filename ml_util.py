#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

def pca_2d(*args):
    """
    Compress data by PCA and return
    :param args: lisr of ndarray
    :return: list of data compressed by PCA
    """

    # check whether args are all ndarray
    for data in args:
        if not isinstance(data, np.ndarray):
            raise Exception('arguments must be list of ndarray.')

    # compress data by PCA
    all_data = np.vstack(args)
    pca = PCA(n_components=2)
    all_data_pca = pca.fit(all_data).transform(all_data)

    # divide data
    result_list = []
    prev_size = 0
    for arg in args:
        size = arg.shape[0]
        data = all_data_pca[prev_size:prev_size + size]
        result_list.append(data)
        prev_size += size

    return result_list

def nearest_neighbor(X, Y):

    min_dist = None
    min_x_idx = None
    min_y_idx = None

    for i, x_row in enumerate(X):
        for j, y_row in enumerate(Y):
            d = distance.euclidean(x_row, y_row)
            if min_dist is None or d < min_dist:
                min_dist = d
                min_x_idx = i
                min_y_idx = j
    return (min_dist, min_x_idx, min_y_idx)
