#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import sys
import os.path
import multiprocessing
import pandas as pd
import h5py

class Word2VecUtil:

    def __init__(self):
        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.model = None
        self.word_vector_df = None

    def learn_word2vec(self, corpus_file, save_file=None, size=200):
        self.logger.info("reading corpus file")
        text8 = word2vec.Text8Corpus(corpus_file)
        self.logger.info("learning word2vec")
        self.model = word2vec.Word2Vec(text8, size=size, window=5, min_count=5, workers=multiprocessing.cpu_count())
        self.logger.info("saving model data")
        if save_file:
            self.model.save(save_file)
        self.logger.info("completed learning word2vec")

    def load_model(self, save_file):
        self.logger.info("loading word2vec model data")
        self.model = word2vec.Word2Vec.load(save_file)
        self.logger.info("completed loading word2vec model data")
        return self.model

    def print_most_similar_words(self, word):
        out = self.model.most_similar(positive=[word])

        for x in out:
            print x[0],x[1]

    def get_most_similar_words_matrix(self, word):
        out = self.model.most_similar(positive=[word])
        matrix = np.array([self.model[word]] + [self.model[x[0]] for x in out])
        labels = np.array([word] + [x[0] for x in out])
        return matrix, labels

    def create_word_features(self, tag_list):
        self.logger.info("creating word features")
        word_vectors = []
        for i, tags in enumerate(tag_list):
            self.logger.info("image id: %d", i + 1)
            for tag in tags:
                word_vectors.append(self.model[tag])
        self.word_vector_mat = np.array(word_vectors)

    def save_word_features(self, filepath):
        self.logger.info("saving word vector as pickle")

        f = h5py.File(filepath, "w")
        f.create_dataset("word_vector_mat", data=self.word_vector_mat)
        f.flush()
        f.close()


    def load_word_features(self, filepath):
        self.logger.info("loading word vector")
        f = h5py.File(filepath, "r")
        self.features_mat = f["word_vector_mat"].value
        f.flush()
        f.close()

    def plot_pca_data(self, X, labels):

        # PCA
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)

        # begin plot
        plt.figure()
        # plot all points(first point is different color)
        plt.scatter(X_r[:, 0], X_r[:, 1], marker = 'o',c = [0 if i == 0 else 1 for i in range(X_r.shape[0])], s = 80, cmap = plt.get_cmap('Spectral'))
        # draw annotations
        for label, x, y in zip(labels, X_r[:, 0], X_r[:, 1]):
            plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.title('PCA')
        plt.show()

    def plot_most_similar_data(self, word):
        X, labels = self.get_most_similar_words_matrix(word)
        self.plot_pca_data(X, labels)

    def plot_most_similar_data2(self, word, word2):
        X, labels = self.get_most_similar_words_matrix(word)
        X2, labels2 = self.get_most_similar_words_matrix(word2)
        X_1_2 = np.r_[X, X2]
        labels_1_2 = np.r_[labels, labels2]
        self.plot_pca_data(X_1_2, labels_1_2)

