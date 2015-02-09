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
        self.cca = None

    def learn_jawiki_corpus(self, save_file):
        self.word2vec.learn_word2vec(self.jawiki_path, save_file)

    def learn_enwiki_corpus(self, save_file):
        self.word2vec.learn_word2vec(self.enwiki_path, save_file)

    def learn_text8_corpus(self, save_file):
        self.word2vec.learn_word2vec('text8', save_file)

    def load_model(self, save_file):
        self.word2vec.load_model(save_file)

    def create_tag_dict(self):
        self.flickr.create_tag_dict(False, self.word2vec.model.vocab.keys())

    def save_flickr_tag_json(self):
        self.flickr.save_tag_dict_as_json(Joint.TAG_DICT_JSON)

    def load_flickr_tag_json(self):
        return self.flickr.load_tag_dict_as_json(Joint.TAG_DICT_JSON)

    def load_features(self):
        self.flickr.load_features(self.feature_path)

    def create_avg_features_df(self):
        return self.flickr.create_avg_features_df()

    def create_word2vec_features_df(self):
        vocab_list = self.flickr.tag_dict.keys()
        return self.word2vec.create_word_vector_df(vocab_list)

    def save_image_features(self):
        self.flickr.save_avg_features_df(Joint.FEATURE_PICKLE)

    def save_word_vector(self):
        self.word2vec.save_word_vector_df(Joint.WORDVECTOR_PICKLE)

    def load_image_features(self):
        self.flickr.load_avg_features_df(Joint.FEATURE_PICKLE)

    def load_word_vector(self):
        self.word2vec.load_word_vector_df(Joint.WORDVECTOR_PICKLE)

    def calc_cca(self, n_components=10, reg_param=0.1):
        self.cca = MyCCA(n_components=n_components, reg_param=reg_param, calc_time=True)
        self.cca.fit(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)

    def save_cca(self):
        f = open(Joint.CCA_PICKLE, 'wb')
        pickle.dumps(self.cca, f)
        f.close()

    def load_cca(self):
        f = open(Joint.CCA_PICKLE, 'rb')
        self.cca = pickle.load(f)
        f.close()

    def calc_cca_transform(self):
        for n in xrange(10, 200, 10):
            self.cca.n_components = n
            x_c, y_c = self.cca.transform(self.word2vec.word_vector_df.values.T, self.flickr.feature_avg_df.values.T)
            np.save(self.tmp_dir_path + 'cca_' + str(n) + 'x.npy', x_c)
            np.save(self.tmp_dir_path + 'cca_' + str(n) + 'y.npy', y_c)
