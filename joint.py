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
        self.wiki_path = self.config['wiki_corpus']['wiki_path']
        self.annotation_path = self.config['flickr_dataset']['annotation_dir_path']
        self.tag_path = self.config['flickr_dataset']['tag_dir_path']
        self.feature_path = self.config['features']['image_feature_path']
        self.tmp_dir_path = self.config['tmp_data']['tmp_dir_path']
        self.model_path = self.config['word2vec_model']['model_path']

        # create object
        self.word2vec = Word2VecUtil()
        self.flickr = FlickrDataSet(self.annotation_path, self.tag_path)
        self.cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)

    def learn_wiki_corpus(self, size=200):
        """
        Learn model by word2vec. The model data is saved after learning.
        :param size: dimension of word vector
        :return: None
        """

        self.word2vec.learn_word2vec(self.wiki_path, self.model_path, size)

    def load_model(self):
        """
        Load word2vec model. Model data is learned by learn_wiki_corpus().
        :return: None
        """

        self.word2vec.load_model(self.model_path)

    def load_flickr_features(self):
        """
        Load flickr features. flickr feature is learned by flickr_img2features.py. the file is in 'https://github.com/rupy/caffe_script/blob/master/flickr_img2features.py'.
        :return: None
        """

        self.flickr.load_raw_features(self.feature_path)

    def create_feature_matrices(self):
        """
        Create image and word feature matrices. Features are saved at the end of calculation.
        :return: None
        """

        # preparation check
        if self.word2vec.model is None:
            Exception('Word2vec model is not learned. You should run learn_wiki_corpus() or load_model(), first.')
        if self.flickr.features_mat is None:
            Exception('feature matrix is not set. You should run load_flickr_features(), first.')

        # calculation
        self.word2vec.create_word_features(self.flickr.tag_list)

        # save
        self.flickr.save_img_features(Joint.FEATURE_PICKLE)
        self.word2vec.save_word_features(Joint.WORDVECTOR_PICKLE)

    def load_feature_matrices(self):
        """
        Load image and word feature matrices. Feature matrices are calculated by create_feature_matrices().
        :return:
        """

        self.flickr.load_avg_features_df(Joint.FEATURE_PICKLE)
        self.word2vec.load_word_vector_df(Joint.WORDVECTOR_PICKLE)

    def fit_data_by_cca(self):
        """
        Learn CCA using image and word features. Learned CCA model is saved at the end of calculation.
        :return: None
        """

        # preparation check
        if self.flickr.feature_avg_df is None:
            Exception('flickr image feature is not set. You should run create_feature_matrices() or load_feature_matrices(), first.')
        if self.word2vec.word_vector_df is None:
            Exception('tag feature is not set. You should run create_feature_matrices() or load_feature_matrices(), first.')

        print self.word2vec.word_vector_df.values.T.shape
        print self.flickr.feature_avg_df.values.T.shape

        # fit
        self.cca.fit(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)

        # save
        self.cca.save_params_as_pickle(self.tmp_dir_path + Joint.CCA_PICKLE)

    def load_cca(self):
        """
        Load learned CCA model. CCA model is calculated by fit_data_by_cca().
        :return:
        """

        self.cca.load_params_from_pickle(self.tmp_dir_path + Joint.CCA_PICKLE)

    def transform_data(self, probabilistic=False):
        """
        Transform feature data by CCA or PCCA changing n_components from 10 to 200. Results of CCA (or PCCA) transformation are saved at the end of calculation.
        :param probabilistic: False if use CCA and True if use PCCA.
        :return: None
        """

        # preparation check
        if self.flickr.feature_avg_df is None:
            Exception('flickr image feature is not set. You should run create_feature_matrices() or load_feature_matrices(), first.')
        if self.word2vec.word_vector_df is None:
            Exception('tag feature is not set. You should run create_feature_matrices() or load_feature_matrices(), first.')
        if self.word2vec.word_vector_df is None:
            Exception('tag feature is not set. You should run create_feature_matrices() or load_feature_matrices(), first.')
        if self.cca.check_fit_finished():
            Exception('cca fit is not finished. You should run fit_cca() or load_cca(), first.')

        # transform and save
        for n in xrange(10, 210, 10):
            if probabilistic:
                self.logger.info("pcca transform: n_components is %d", n)
                self.cca.n_components = n
                x_c, y_c, z = self.cca.ptransform(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)
                np.save(self.tmp_dir_path + 'pcca_' + str(n) + 'x.npy', x_c)
                np.save(self.tmp_dir_path + 'pcca_' + str(n) + 'y.npy', y_c)
                np.save(self.tmp_dir_path + 'pcca_' + str(n) + 'z.npy', z)
            else:
                self.logger.info("cca transform: n_components is %d", n)
                self.cca.n_components = n
                x_c, y_c = self.cca.transform(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)
                np.save(self.tmp_dir_path + 'cca_' + str(n) + 'x.npy', x_c)
                np.save(self.tmp_dir_path + 'cca_' + str(n) + 'y.npy', y_c)

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
        if probabilistic:
            x_c = np.load(self.tmp_dir_path + 'pcca_' + str(n_components) + 'x.npy')
            y_c = np.load(self.tmp_dir_path + 'pcca_' + str(n_components) + 'y.npy')
            z = np.load(self.tmp_dir_path + 'pcca_' + str(n_components) + 'z.npy')
            self.cca.X_pc = x_c
            self.cca.Y_pc = y_c
            self.cca.Z_pc = z
        else:
            x_c = np.load(self.tmp_dir_path + 'cca_' + str(n_components) + 'x.npy')
            y_c = np.load(self.tmp_dir_path + 'cca_' + str(n_components) + 'y.npy')
            self.cca.X_c = x_c
            self.cca.Y_c = y_c

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