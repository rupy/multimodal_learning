#!/usr/bin/python
#-*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

class PCASpace:

    def __init__(self):
        self.pca = PCA(n_components=2)

    def fit(self, *args):
        # check whether args are all ndarray
        for data in args:
            if not isinstance(data, np.ndarray):
                raise Exception('arguments must be list of ndarray.')

        # compress data by PCA
        all_data = np.vstack(args)
        self.pca.fit(all_data)

    def transform(self, data):
        data_t = self.pca.transform(data)
        return data_t

    def __plot_points_with_labels(self, data, labels):

        data_t = self.transform(data)
        # plot all points
        plt.plot(data_t[:, 0], data_t[:, 1], "xb")
        # draw annotations
        for label, x, y in zip(labels, data_t[:, 0], data_t[:, 1]):
            plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    def __plot_points_with_img(self, data, images):

        data_t = self.transform(data)

        # plot all points
        plt.plot(data_t[:, 0], data_t[:, 1], ".r")
        # draw images
        ax = plt.gcf().gca()
        for img, x, y in zip(images, data_t[:, 0], data_t[:, 1]):

            # add a first image
            imagebox = OffsetImage(img, zoom=.1)
            xy = [x, y]               # coordinates to position this image

            ab = AnnotationBbox(imagebox, xy,
                # xybox=(arr_hand.shape[1] * 0.1 / 2, -arr_hand.shape[0] * 0.1 / 2),
                xybox=(0, 0),
                xycoords='data',
                boxcoords="offset points"
            )
            ab.set_zorder(0)
            ax.add_artist(ab)

    def plot_data_with_tag_img(self, X, x_labels, Y, y_img_paths):

        images = [plt.imread(img) for img in y_img_paths]
        print images[0]
        # begin plot
        plt.figure()
        self.__plot_points_with_img(Y, images)
        self.__plot_points_with_labels(X, x_labels)
        plt.title('PCA')
        plt.show()
