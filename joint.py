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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
import ml_util as ml
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from pca_space import PCASpace

class Joint:

    CONFIG_YAML = 'config.yml'
    FEATURE_SAVE_FILE ='features/image_feature.npy'
    WORDVECTOR__SAVE_FILE ='features/word_vector.npy'
    TAG_LIST_SAVE_FILE = 'tmp/tag_list.npy'
    IMG_LABEL_SAVE_FILE = 'tmp/img_label.npy'
    TAG_LABEL_SAVE_FILE = 'tmp/tag_label.npy'
    CCA_PARAMS_SAVE_DIR = 'cca_params/'

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
        self.pca_space = PCASpace()

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
        self.flickr.save_tag_list(self.output_dir_path + Joint.TAG_LIST_SAVE_FILE)

    def create_labels(self):
        """
        Create tag & image labels for features. labels are saved at the end of calculation.
        :return: None
        """
        self.flickr.load_tag_list(self.output_dir_path + Joint.TAG_LIST_SAVE_FILE)

        # create labels from tag list
        self.flickr.create_img_label()
        self.flickr.create_tag_label()

        # save
        self.flickr.save_tag_label(self.output_dir_path + Joint.TAG_LABEL_SAVE_FILE)
        self.flickr.save_img_label(self.output_dir_path + Joint.IMG_LABEL_SAVE_FILE)

    def create_image_feature_matrix(self):
        """
        Create image and word feature matrices. Features are saved at the end of calculation.
        :return: None
        """

        self.flickr.load_tag_list(self.output_dir_path + Joint.TAG_LIST_SAVE_FILE)

        # calculation
        self.flickr.load_raw_features(self.original_feature_path)
        self.flickr.save_img_features(self.output_dir_path + Joint.FEATURE_SAVE_FILE)


    def create_word_feature_matrix(self):
        """
        Create image feature matrices. Features are saved at the end of calculation.
        :return: None
        """
        self.word2vec.load_model(self.model_path)
        tag_list = self.flickr.load_tag_list(self.output_dir_path + Joint.TAG_LIST_SAVE_FILE)

        self.word2vec.create_word_features(tag_list)
        self.word2vec.save_word_features(self.output_dir_path + Joint.WORDVECTOR__SAVE_FILE)

    def fit_data_by_cca(self):
        """
        Learn CCA using image and word features. Learned CCA model is saved at the end of calculation.
        :return: None
        """

        # preparation
        self.flickr.load_img_features(self.output_dir_path + Joint.FEATURE_SAVE_FILE)
        self.logger.info("features_mat shape is %s", self.flickr.features_mat.shape)
        self.word2vec.load_word_features(self.output_dir_path + Joint.WORDVECTOR__SAVE_FILE)
        self.logger.info("word_vector_mat shape is %s", self.word2vec.word_vector_mat.shape)

        # fit
        self.cca.fit(self.word2vec.word_vector_mat, self.flickr.features_mat)

        # save
        self.cca.save_params_as_pickle(self.output_dir_path + Joint.CCA_PARAMS_SAVE_DIR)

    def plot_original_data(self):
        """
        plot original two data.
        :return: None
        """

        # preparation
        self.flickr.load_img_features(self.output_dir_path + Joint.FEATURE_SAVE_FILE)
        self.logger.info("features_mat shape is %s", self.flickr.features_mat.shape)
        self.word2vec.load_word_features(self.output_dir_path + Joint.WORDVECTOR__SAVE_FILE)
        self.logger.info("word_vector_mat shape is %s", self.word2vec.word_vector_mat.shape)

        # plot
        self.cca.plot_original_data(self.word2vec.word_vector_mat, self.flickr.features_mat)

    def transform_data(self):
        """
        Transform feature data by CCA(PCCA). Results of CCA (or PCCA) transformation are saved at the end of calculation.
        :param probabilistic: False if use CCA and True if use PCCA.
        :return: None
        """

        # preparation
        self.flickr.load_img_features(self.output_dir_path + Joint.FEATURE_SAVE_FILE)
        self.logger.info("features_mat shape is %s", self.flickr.features_mat.shape)
        self.word2vec.load_word_features(self.output_dir_path + Joint.WORDVECTOR__SAVE_FILE)
        self.logger.info("word_vector_mat shape is %s", self.word2vec.word_vector_mat.shape)
        self.cca.load_params_from_pickle(self.output_dir_path + Joint.CCA_PARAMS_SAVE_DIR)

        # transform and save
        x_c, y_c, z_c = self.cca.ptransform(self.word2vec.word_vector_mat, self.flickr.features_mat)
        np.save(self.output_dir_path + 'cca/cca_x.npy', x_c)
        np.save(self.output_dir_path + 'cca/cca_y.npy', y_c)
        np.save(self.output_dir_path + 'cca/cca_z.npy', z_c)

        self.cca.fix_reverse()


    def load_transformed_data(self, probabilistic=True, n_components=200):
        """
        Load transfromed feature data by CCA. the data is calculated by fit_data_by_cca().
        :param probabilistic: False if use CCA and True if use PCCA.
        :param n_components: dimension of transformed data to load.
        :return: None
        """

        self.cca.n_components = n_components
        x_c = np.load(self.output_dir_path + 'cca/cca_x.npy')
        y_c = np.load(self.output_dir_path + 'cca/cca_y.npy')
        self.cca.X_c = x_c
        self.cca.Y_c = y_c
        z_c = None
        if probabilistic:
            z_c = np.load(self.output_dir_path + 'cca/cca_z.npy')
            self.cca.Z_c = z_c

        self.cca.fix_reverse()

        return x_c, y_c, z_c

    def plot_transformed_data(self, probabilistic=True):
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

    def load_labels(self):
        self.flickr.load_tag_label(self.output_dir_path + Joint.TAG_LABEL_SAVE_FILE)
        self.flickr.load_img_label(self.output_dir_path + Joint.IMG_LABEL_SAVE_FILE)

    def pca_fit(self, n_components=10):
        all_data = np.vstack([self.cca.X_c[:, 0:n_components], self.cca.X_c[:, 0:n_components]])
        self.pca_space.fit(all_data)

    def plot_tag_data(self, search_tag, n_components=10):
        X_c = self.cca.X_c[:, 0:n_components]

        # tag-img pair is not only one pair
        indices = [idx for idx, tag in enumerate(self.flickr.tag_label) if tag == search_tag ]
        indices_not = [idx for idx, tag in enumerate(self.flickr.tag_label) if tag != search_tag]

        avg = stats.trim_mean(X_c[indices], 0.1)
        ml.plot_3_data(X_c[indices_not], ml.add_jitter(X_c[indices]), avg )

    def plot_tag_img_pairs(self, search_tag, n_components=10):
        X_c = self.cca.X_c[:, 0:n_components]
        Y_s = self.cca.Y_s[:, 0:n_components]

        # tag-img pair is not only one pair
        indices = [idx for idx, tag in enumerate(self.flickr.tag_label) if tag == search_tag ]

        ml.plot_2_data(ml.add_jitter(X_c[indices]), ml.add_jitter(Y_s[indices]))

    def tag_nearest_neighbor(self, search_tag, n_components=10):

        X_c = self.cca.X_c[:, 0:n_components]
        Y_s = self.cca.Y_s[:, 0:n_components]

        # tag-img pair is not only one pair
        indices = [idx for idx, tag in enumerate(self.flickr.tag_label) if tag == search_tag ]
        # calc mean because tag is not unique
        tag_data_mean = stats.trim_mean(X_c[indices], 0.1)

        # calc nearest neighbors
        nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Y_s)
        dists, nn_indices = nn.kneighbors([tag_data_mean], 200, return_distance=True)

        # transform image feature indices to dataset indices
        nn_dataset_indices_dup = [self.flickr.img_label[idx] + 1 for idx in nn_indices[0]]
        nn_dataset_indices = list(sorted(set(nn_dataset_indices_dup), key=nn_dataset_indices_dup.index))

        # plot nearest images
        for dataset_idx in nn_dataset_indices:
            self.flickr.plot_image_with_tags_by_id(dataset_idx)

        # ml.plot_data_with_img(Y_s[nn_dataset_indices], nn_dataset_indices, np.array(tag_data_mean), [search_tag])
        dataset_paths = [self.flickr.dataset_dir_path + "im%d.jpg" % idx for idx in nn_dataset_indices]
        self.pca_space.plot_data_with_tag_img(np.array(tag_data_mean), [search_tag], Y_s[nn_dataset_indices], dataset_paths)



    def plot_img_data(self, search_dataset_id, n_components=10):
        Y_c = self.cca.Y_c[:, 0:n_components]

        # tag-img pair is not only one pair
        indices = [idx for idx, dataset_idx in enumerate(self.flickr.img_label) if dataset_idx == search_dataset_id - 1 ]
        indices_not = [idx for idx, dataset_idx in enumerate(self.flickr.tag_label) if dataset_idx != search_dataset_id - 1]

        ml.plot_3_data(Y_c[indices_not], ml.add_jitter(Y_c[indices]),Y_c[indices[0]] )

    def img_nearest_neighbor(self, search_dataset_id, n_components=10):

        X_c = self.cca.X_c[:, 0:n_components]
        Y_s = self.cca.Y_s[:, 0:n_components]

        # get feature index from dataset index
        feature_idx = self.flickr.img_label.tolist().index(search_dataset_id - 1 )

        # calc nearest neighbors
        nn = NearestNeighbors(n_neighbors=50).fit(X_c)
        dists, nn_indices = nn.kneighbors([Y_s[feature_idx]], 50, return_distance=True)
        print dists
        print nn_indices

        # transform image feature indices to dataset indices
        nn_tags_dup = [self.flickr.tag_label[idx] for idx in nn_indices[0]]
        nn_tags = list(sorted(set(nn_tags_dup), key=nn_tags_dup.index))
        # show tags & plot image
        print nn_tags
        self.flickr.plot_img_by_id(search_dataset_id)

        self.pca_space.plot_data_with_tag_img(
            X_c[nn_indices[0]],
            self.flickr.tag_label[nn_indices][0],
            np.array([Y_s[feature_idx]]),
            [self.flickr.dataset_dir_path + "im%d.jpg" % search_dataset_id]
        )

    def plot_img_by_tag(self, tag):
        self.flickr.load_tag_list(self.output_dir_path + Joint.TAG_LIST_SAVE_FILE)
        self.flickr.plot_images_by_tag(tag)

    def plot_points(self, data, xmin=None, xmax=None, ymin=None, ymax=None):
        print data.shape
        plt.plot(data[:, 0], data[:, 1], 'xb')
        if xmin and xmax:
            plt.xlim(xmin, xmax)
        if ymin and ymax:
            plt.ylim(ymin, ymax)
        plt.show()

    def plot_points_2(self, X, Y, data,  xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(X[:, 0], X[:, 1], 'xb')
        plt.plot(Y[:, 0], Y[:, 1], '.r')
        plt.plot(data[:, 0], data[:, 1], 'og')
        if xmin and xmax:
            plt.xlim(xmin - (xmax - xmin) * 50, xmax + (xmax - xmin) * 50)
        if ymin and ymax:
            plt.ylim(ymin - (ymax - ymin) * 50, ymax + (ymax - ymin) * 50)
        plt.show()

    def plot_points_3(self, X, Y, min_tag_idx, min_img_idx):

        X_r, Y_r = ml.pca_2d(X, Y)

        min_d, min_idx_y = ml.nearest_neighbor(X_r[min_tag_idx], Y_r)

        dataset_idx = self.flickr.img_label[min_idx_y] + 1
        self.flickr.plot_image_with_tags_by_id(dataset_idx)

        plt.plot(X_r[:, 0], X_r[:, 1], 'xb')
        plt.plot(Y_r[:, 0], Y_r[:, 1], '.r')
        data = np.array([X_r[min_tag_idx], Y_r[min_img_idx]])
        print distance.euclidean(data[0], data[1])
        plt.plot(data[:, 0], data[:, 1], 'og')

        plt.show()

    def plot_img_in_plot(self, img_id):
        ml.plot_data_with_img(np.array([[1,1], [2,2]]), [1,2],np.array([[1.5,1.5], [1.5,3.0]]), ["a","b"])

if __name__=="__main__":

    logging.root.setLevel(level=logging.INFO)

    joint = Joint()
    save_file = 'enwiki_word2vec_200dim.dat'
    joint.word2vec.load_model(save_file)
    print joint.word2vec.model[u'mother']