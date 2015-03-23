#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt

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

def nearest_neighbor(search_point, data):

    min_dist = None
    min_idx = None
    min_indices = []

    for i, data_point in enumerate(data):
        d = distance.euclidean(search_point, data_point)
        if min_dist is None or d < min_dist:
            min_dist = d
            min_idx = i
            min_indices.append(min_idx)
    return (min_dist, min_idx, min_indices)

def plot_data_2d(X, Y):
    X_r, Y_r = pca_2d(X, Y)
    print X_r.shape
    print Y_r.shape
    plt.plot(X_r[:, 0], X_r[:, 1], 'xb')
    plt.plot(Y_r[:, 0], Y_r[:, 1], 'or')
    plt.show()

def plot_data_3d(X, Y, Z):
    X_r, Y_r, Z_r = pca_2d(X, Y, Z)
    plt.plot(X_r[:, 0], X_r[:, 1], 'xb')
    plt.plot(Y_r[:, 0], Y_r[:, 1], '.r')
    plt.plot(Z_r[:, 0], Z_r[:, 1], 'og')
    plt.show()

def add_jitter(data, loc=0.0, scale=0.05):
    return data + np.random.normal(loc=loc, scale=scale, size=data.shape)