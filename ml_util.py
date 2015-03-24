#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

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
    plt.plot(Y_r[:, 0], Y_r[:, 1], '.g')
    plt.show()

def plot_data_3d(X, Y, Z):
    X_r, Y_r, Z_r = pca_2d(X, Y, Z)
    plt.plot(X_r[:, 0], X_r[:, 1], 'xb')
    plt.plot(Y_r[:, 0], Y_r[:, 1], '.g')
    plt.plot(Z_r[:, 0], Z_r[:, 1], 'or')
    plt.show()

def add_jitter(data, loc=0.0, scale=0.05):
    return data + np.random.normal(loc=loc, scale=scale, size=data.shape)

def plot_data_with_tags(X, Y, x_labels, y_labels):

    # PCA
    X_r, Y_r = pca_2d(X, Y)

    # begin plot
    plt.figure()
    # plot all points(first point is different color)
    plt.plot(X_r[:, 0], X_r[:, 1], "xb")
    plt.plot(Y_r[:, 0], Y_r[:, 1], "or")
    # draw annotations
    for label, x, y in zip(x_labels, X_r[:, 0], X_r[:, 1]):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    for label, x, y in zip(y_labels, Y_r[:, 0], Y_r[:, 1]):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.title('PCA')
    plt.show()

def plot_data_with_img(X, x_img_labels, Y, y_tag_labels):

    # PCA
    X_r, Y_r = pca_2d(X, Y)

    # begin plot
    # plot all points(first point is different color)
    plt.plot(X_r[:, 0], X_r[:, 1], "xb")
    plt.plot(Y_r[:, 0], Y_r[:, 1], "ob")
    # draw images
    for label, x, y in zip(x_img_labels, X_r[:, 0], X_r[:, 1]):

        # add a first image
        arr_hand = plt.imread('/Users/rupy/dataset/mirflickr/im%d.jpg' % label)
        imagebox = OffsetImage(arr_hand, zoom=.1)
        xy = [x, y]               # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
            # xybox=(arr_hand.shape[1] * 0.1 / 2, -arr_hand.shape[0] * 0.1 / 2),
            xybox=(0, 0),
            xycoords='data',
            boxcoords="offset points")
        plt.gcf().gca().add_artist(ab)

    for label, x, y in zip(y_tag_labels, Y_r[:, 0], Y_r[:, 1]):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.grid(True)
    plt.title('PCA')
    plt.show()
