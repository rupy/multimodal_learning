#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

from word2vec_util import Word2VecUtil
from flickr_data_set import FlickrDataSet
import logging
import os
import sys
import yaml
from mycca import MyCCA
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Joint:

    CONFIG_YAML = 'config.yml'
    TAG_DICT_JSON ='tags_dict.json'
    FEATURE_PICKLE ='image_feature.pkl'
    WORDVECTOR_PICKLE ='word_vector.pkl'
    CCA_PICKLE = 'cca.pkl'

    def __init__(self):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # load config file
        f = open(Joint.CONFIG_YAML, 'r')
        self.config = yaml.load(f)
        f.close()
        self.jawiki_path = self.config['wiki_corpus']['jawiki_path']
        self.enwiki_path = self.config['wiki_corpus']['enwiki_path']
        self.text8_path = self.config['wiki_corpus']['text8_path']
        self.annotation_path = self.config['flickr_dataset']['annotation_dir_path']
        self.tag_path = self.config['flickr_dataset']['tag_dir_path']
        self.feature_path = self.config['features']['image_feature_path']
        self.tmp_dir_path = self.config['tmp_data']['tmp_dir_path']

        # create object
        self.word2vec = Word2VecUtil()
        self.flickr = FlickrDataSet(self.annotation_path, self.tag_path)
        self.cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)

    def learn_jawiki_corpus(self, save_file):
        self.word2vec.learn_word2vec(self.jawiki_path, save_file)

    def learn_enwiki_corpus(self, save_file):
        self.word2vec.learn_word2vec(self.enwiki_path, save_file)

    def learn_text8_corpus(self, save_file):
        self.word2vec.learn_word2vec(self.text8_path, save_file)

    def create_tag_dict(self, save_file):
        self.word2vec.load_model(save_file)
        self.flickr.create_tag_dict(False, self.word2vec.model.vocab.keys())
        self.flickr.save_tag_dict_as_json(Joint.TAG_DICT_JSON)

    def create_feature_matrix(self, save_file):
        self.word2vec.load_model(save_file)
        self.flickr.load_tag_dict_as_json(Joint.TAG_DICT_JSON)
        self.flickr.load_features(self.feature_path)

        self.flickr.create_avg_features_df()
        vocab_list = self.flickr.tag_dict.keys()
        self.word2vec.create_word_vector_df(vocab_list)
        self.flickr.save_avg_features_df(Joint.FEATURE_PICKLE)
        self.word2vec.save_word_vector_df(Joint.WORDVECTOR_PICKLE)

    def calc_cca(self):
        self.flickr.load_avg_features_df(Joint.FEATURE_PICKLE)
        self.word2vec.load_word_vector_df(Joint.WORDVECTOR_PICKLE)
        self.cca.fit(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)
        self.cca.save_params_as_pickle(self.tmp_dir_path + Joint.CCA_PICKLE)
        self.__cca_transform_and_save()

    def load_and_calc_cca(self):
        self.flickr.load_avg_features_df(Joint.FEATURE_PICKLE)
        self.word2vec.load_word_vector_df(Joint.WORDVECTOR_PICKLE)
        self.cca.load_params_from_pickle(self.tmp_dir_path + Joint.CCA_PICKLE)
        self.__cca_transform_and_save()

    def load_cca_result(self, n_components=200):
        x_c = np.load(self.tmp_dir_path + 'cca_' + str(n_components) + 'x.npy')
        y_c = np.load(self.tmp_dir_path + 'cca_' + str(n_components) + 'y.npy')
        return x_c, y_c

    def __cca_transform_and_save(self):
        for n in xrange(10, 210, 10):
            self.logger.info("cca transform: n_components is %d", n)
            self.cca.n_components = n
            x_c, y_c = self.cca.transform(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)
            np.save(self.tmp_dir_path + 'cca_' + str(n) + 'x.npy', x_c)
            np.save(self.tmp_dir_path + 'cca_' + str(n) + 'y.npy', y_c)

    def plot_cca(self, X, Y):
        # PCA
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        Y_r = pca.fit(Y).transform(Y)

        # begin plot
        plt.figure()

        plt.subplot(221)
        plt.plot(Y_r[:, 0], Y_r[:, 1], 'xb')
        plt.plot(X_r[:, 0], X_r[:, 1], '.r')
        plt.title('PCA - CCA XY')

        plt.subplot(222)
        plt.plot(X_r[:, 0], X_r[:, 1], '.r')
        plt.title('PCA - CCA X')

        plt.subplot(223)
        plt.plot(Y_r[:, 0], Y_r[:, 1], 'xb')
        plt.title('PCA - CCA Y')

        plt.show()
