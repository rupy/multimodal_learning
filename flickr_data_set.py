#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import pandas as pd
import sys
import logging
from collections import defaultdict
try:
   import cPickle as pickle
except:
   import pickle

class FlickrDataSet:

    DATASET_SIZE = 25000

    def __init__(self, annotation_dir_path, dataset_dir_path):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.annotation_dir_path = annotation_dir_path
        self.dataset_dir_path = dataset_dir_path
        self.annotation_df = None
        self.tag_list = []
        self.tag_label = []
        self.img_label = []

        self.features_mat = None
        self.feature_avg_df = None

    """
    annotation
    """
    def __get_ids_from_annotation_file(self, file_name):

        f = open(self.annotation_dir_path + file_name)
        ids = f.read().strip().split('\r\n')
        ids = map(int, ids)
        f.close()
        return ids

    def create_annotation_df(self, relevant=False):

        # get annotation file list
        files = os.listdir(self.annotation_dir_path)
        annotation_files = None
        if relevant:
            annotation_files = [annotation_file for annotation_file in files if annotation_file != 'README.txt' and annotation_file.endswith("r1.txt")]
        else:
            annotation_files = [annotation_file for annotation_file in files if annotation_file != 'README.txt' and not annotation_file.endswith("r1.txt")]

        # create annotation list
        annotation_list = [set() for i in xrange(FlickrDataSet.DATASET_SIZE)]
        for file_name in annotation_files:
            ids = self.__get_ids_from_annotation_file(file_name)
            for img_id in ids:
                annotation_list[img_id - 1].add(os.path.splitext(file_name)[0])

        # convert annotation list to class matrix
        all_annotations = map(lambda f:os.path.splitext(f)[0], annotation_files)
        annotation_matrix = np.array(
            # generate class vector for each id
            [[ 1 if all_annotations[i] in annotations else 0 for i in xrange(len(all_annotations))] for annotations in annotation_list]
        )

        # convert matrix to DataFrame and store
        self.annotation_df = pd.DataFrame(annotation_matrix, pd.Series([i + 1 for i in xrange(FlickrDataSet.DATASET_SIZE)]), pd.Series(all_annotations))

    def get_all_annotations(self):

        return self.annotation_df.columns.values

    def get_annotations_by_id(self, img_id):

        return self.annotation_df.columns[self.annotation_df.ix[img_id] == 1].values

    def get_ids_by_annotation(self, class_name):

        return self.annotation_df.index[self.annotation_df[class_name] == 1].values

    """
    tags
    """
    def plot_img_by_id(self, img_id):

        file_name = 'im%d.jpg' % img_id
        img=mpimg.imread(self.dataset_dir_path + file_name)
        plt.imshow(img)
        plt.show()

    def __get_tags_from_tag_file_by_id(self, img_id, tags_raw=False):

        file_name = 'tags%d.txt' % img_id
        tag_path = (self.dataset_dir_path + 'meta/tags_raw/' + file_name) if tags_raw else (self.dataset_dir_path + 'meta/tags/' + file_name)
        f = open(tag_path)
        tags = f.read().strip().split('\r\n')
        f.close()
        return tags

    def plot_image_with_tags_by_id(self, img_id):

        file_name = 'tags%d.txt' % img_id
        print self.__get_tags_from_tag_file_by_id(img_id)
        self.plot_img_by_id(img_id)

    def plot_images_by_tag(self, tag):
        indices = []
        for idx, tags in enumerate(self.tag_list):
            if tag in tags:
                indices.append(idx)
                self.plot_image_with_tags_by_id(idx + 1)

    def create_tag_list(self, vocab_set=[], tags_raw=False):
        self.logger.info("creating tag list")
        for i in xrange(FlickrDataSet.DATASET_SIZE):
            # open tag file & create vocabulary list
            img_id = i + 1
            file_name = 'tags%d.txt' % img_id
            tag_path = (self.dataset_dir_path + 'meta/tags_raw/' + file_name) if tags_raw else (self.dataset_dir_path + 'meta/tags/' + file_name)
            f = open(tag_path)
            tags = set(f.read().strip().split('\r\n'))
            f.close()
            tags_vocab = tags & set(vocab_set) # check tag if tag is in vocabulary
            self.tag_list.append(tags_vocab)
            self.logger.info("Image ID: %d / %d %d%% %s" % (img_id, FlickrDataSet.DATASET_SIZE, img_id * 100 / FlickrDataSet.DATASET_SIZE, tags_vocab))
        return self.tag_list

    def create_tag_label(self):
        self.logger.info("creating tag label")
        for tags in self.tag_list:
            for tag in tags:
                self.tag_label.append(tag)

    def save_tag_label(self, filepath):
        self.logger.info("saving tag label to %s", filepath)
        np.save(filepath, self.tag_label)

    def load_tag_label(self, filepath):
        self.logger.info("loading tag label to %s", filepath)
        self.tag_label = np.load(filepath)
        return self.tag_label

    def create_img_label(self):
        self.logger.info("creating img label")
        for i, tags in enumerate(self.tag_list):
            for tag in tags:
                self.img_label.append(i)
    def save_img_label(self, filepath):
        self.logger.info("saving img label to %s", filepath)
        np.save(filepath, self.img_label)

    def load_img_label(self, filepath):
        self.logger.info("loading img label to %s", filepath)
        self.img_label = np.load(filepath)
        return self.img_label

    def save_tag_list(self, filepath):
        self.logger.info("saving tag list to %s", filepath)
        np.save(filepath, self.tag_list)

    def load_tag_list(self, filepath):
        self.logger.info("loading tag list from %s", filepath)
        self.tag_list = np.load(filepath)
        return self.tag_list

    def load_raw_features(self, feature_path):
        self.logger.info("loading raw features from %s", feature_path)
        features_mat = np.load(feature_path)
        self.logger.info("feature matrix is %s", features_mat.shape)
        # copy rows
        self.logger.info("creating image feature matrix")
        img_features = []
        for tag_idx, tags in enumerate(self.tag_list):
            self.logger.info("image id: %d", tag_idx + 1)
            for tag in tags:
                img_features.append(features_mat[tag_idx])
        self.features_mat = np.array(img_features)
        self.logger.info("created image feature matrix as %s", self.features_mat.shape)
        return self.features_mat

    def save_img_features(self, filepath):
        self.logger.info("saving features")
        np.save(filepath, self.features_mat)

    def load_img_features(self, filepath):
        self.logger.info("loading features")
        self.features_mat = np.load(filepath)

