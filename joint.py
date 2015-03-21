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
import h5py

class Joint:

    CONFIG_YAML = 'config.yml'
    FEATURE_SAVE_FILE ='features/image_feature.npy'
    WORDVECTOR__SAVE_FILE ='features/word_vector.npy'
    TAG_LIST_SAVE_FILE = 'tmp/tag_list.pkl'
    CCA_SAVE_FILE = 'cca/cca.pkl'
    HDF5_SAVE_DATA = 'save.h5'

    def __init__(self):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # load config file
        f = open(Joint.CONFIG_YAML, 'r')
        self.config = yaml.load(f)
        f.close()
        self.wiki_path = self.config['wiki_corpus']['wiki_path']
        self.annotation_path = self.config['flickr_dataset']['annotation_dir_path']
        self.dataset_path = self.config['flickr_dataset']['dataset_dir_path']
        self.original_feature_path = self.config['original_feature']['image_raw_feature_path']
        self.output_dir_path = self.config['output']['output_dir_path']
        self.model_path = self.config['word2vec_model']['model_path']

        if not os.path.isfile(self.wiki_path):
            Exception('Cannot find wikipedia corpus in %s' % self.wiki_path)
        if not os.path.isfile(self.annotation_path):
            Exception('Cannot find annotation file in %s' % self.annotation_path)
        if not os.path.isfile(self.dataset_path):
            Exception('Cannot find image dir in %s' % self.dataset_path)
        if not os.path.isfile(self.original_feature_path):
            Exception('Cannot find feature file in %s' % self.original_feature_path)

        # create object
        self.word2vec = Word2VecUtil()
        self.flickr = FlickrDataSet(self.annotation_path, self.dataset_path)
        self.cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)

    def learn_wiki_corpus(self, size=200):
        """
        Learn model by word2vec. The model data is saved after learning.
        :param size: dimension of word vector
        :return: None
        """

        self.word2vec.learn_word2vec(self.wiki_path, self.model_path, size)

    def create_tag_list(self):
        """
        Create tag list that limited by word2vec vocabulary list. Tag list is saved at the end of calculation.
        :return: None
        """
        self.word2vec.load_model(self.model_path)
        self.flickr.create_tag_list(self.word2vec.model.vocab.keys())
        self.flickr.save_tag_list(self.output_dir_path + Joint.HDF5_SAVE_DATA)

    def create_image_feature_matrix(self):
        """
        Create image and word feature matrices. Features are saved at the end of calculation.
        :return: None
        """

        self.flickr.load_tag_list(self.output_dir_path + Joint.HDF5_SAVE_DATA)

        # calculation
        self.flickr.load_raw_features(self.original_feature_path)
        self.flickr.save_img_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)

    def create_word_feature_matrix(self):
        """
        Create image feature matrices. Features are saved at the end of calculation.
        :return: None
        """
        self.word2vec.load_model(self.model_path)
        tag_list = self.flickr.load_tag_list(self.output_dir_path + Joint.HDF5_SAVE_DATA)

        self.word2vec.create_word_features(tag_list)
        self.word2vec.save_word_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)

    def fit_data_by_cca(self):
        """
        Learn CCA using image and word features. Learned CCA model is saved at the end of calculation.
        :return: None
        """

        # preparation
        self.flickr.load_img_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)
        self.logger.info("features_mat shape is %s", self.flickr.features_mat.shape)
        self.word2vec.load_word_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)
        self.logger.info("word_vector_mat shape is %s", self.word2vec.word_vector_mat.shape)

        # fit
        self.cca.fit(self.word2vec.word_vector_mat, self.flickr.features_mat)

        # save
        self.cca.save_params_as_pickle(self.output_dir_path + Joint.CCA_SAVE_FILE)

    def transform_data(self, probabilistic=False):
        """
        Transform feature data by CCA or PCCA changing n_components from 10 to 200. Results of CCA (or PCCA) transformation are saved at the end of calculation.
        :param probabilistic: False if use CCA and True if use PCCA.
        :return: None
        """

        # preparation
        self.flickr.load_img_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)
        self.logger.info("features_mat shape is %s", self.flickr.features_mat.shape)
        self.word2vec.load_word_features(self.output_dir_path + Joint.HDF5_SAVE_DATA)
        self.logger.info("word_vector_mat shape is %s", self.word2vec.word_vector_mat.shape)
        self.cca.load_params_from_pickle(self.output_dir_path + Joint.CCA_SAVE_FILE)

        # transform and save
        f = h5py.File(self.output_dir_path + Joint.HDF5_SAVE_DATA, "w")
        for n in xrange(10, 210, 10):
            if probabilistic:
                self.logger.info("pcca transform: n_components is %d", n)
                self.cca.n_components = n

                x_c, y_c, z = self.cca.ptransform(self.word2vec.word_vector_mat, self.flickr.features_mat)
                f.create_dataset("pcca_"  + str(n) + "x" , data=x_c)
                f.create_dataset("pcca_"  + str(n) + "y" , data=y_c)
                f.create_dataset("pcca_"  + str(n) + "z" , data=z)
            else:
                self.logger.info("cca transform: n_components is %d", n)
                self.cca.n_components = n
                x_c, y_c = self.cca.transform(self.word2vec.word_vector_mat, self.flickr.features_mat)
                f.create_dataset("cca_"  + str(n) + "x" , data=x_c)
                f.create_dataset("cca_"  + str(n) + "y" , data=y_c)
        f.flush()
        f.close()

    def load_transformed_data(self, probabilistic=False, n_components=200):
        """
        Load transfromed feature data by CCA. the data is calculated by fit_data_by_cca().
        :param probabilistic: False if use CCA and True if use PCCA.
        :param n_components: dimension of transformed data to load.
        :return: None
        """

        x_c = None
        y_c = None
        z = None
        f = h5py.File(self.output_dir_path + Joint.HDF5_SAVE_DATA, "r")
        if probabilistic:

            x_c = f['cca/pcca_' + str(n_components) + 'x'].value
            y_c = f['cca/pcca_' + str(n_components) + 'y'].value
            z = f['cca/pcca_' + str(n_components) + 'z'].value
            self.cca.X_pc = x_c
            self.cca.Y_pc = y_c
            self.cca.Z_pc = z
        else:
            x_c = f['cca/cca_' + str(n_components) + 'x'].value
            y_c = f['cca/cca_' + str(n_components) + 'y'].value
            self.cca.X_c = x_c
            self.cca.Y_c = y_c
        f.flush()
        f.close()

        return x_c, y_c, z

    def plot_transformed_data(self, probabilistic=False):
        """
        Plot transformed data by CCA. The data is compressed by PCA and plotted.
        :param probabilistic: False if use CCA and True if use PCCA.
        :return: None
        """

        self.cca.plot_cca_result(probabilistic)

    def print_corrcoef(self):
        """
        print correlation coefficients of result of CCA.
        :return: None
        """

        print self.cca.corrcoef()

if __name__=="__main__":

    logging.root.setLevel(level=logging.INFO)

    joint = Joint()
    save_file = 'enwiki_word2vec_200dim.dat'
    joint.word2vec.load_model(save_file)
    print joint.word2vec.model[u'mother']