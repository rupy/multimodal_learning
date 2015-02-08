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
import json

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
        self.tag_dict = defaultdict(list)

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

    def create_tag_dict(self, tags_raw=False, vocab_set=None):
        self.logger.info("creating tag dictionary")
        for i in xrange(FlickrDataSet.DATASET_SIZE):
            img_id = i + 1
            file_name = 'tags%d.txt' % img_id
            tag_path = (self.dataset_dir_path + 'meta/tags_raw/' + file_name) if tags_raw else (self.dataset_dir_path + 'meta/tags/' + file_name)
            f = open(tag_path)
            tags = set(f.read().strip().split('\r\n'))
            tags_vocab = tags & set(vocab_set)
            f.close()
            for tag in tags_vocab:
                self.tag_dict[tag].append(img_id)
            self.logger.info("Image ID: %d / %d %d%% %s" % (img_id, FlickrDataSet.DATASET_SIZE, img_id * 100 / FlickrDataSet.DATASET_SIZE, tags_vocab))
        return self.tag_dict

    def save_tag_dict_as_json(self, filename):
        self.logger.info("saving tag dictionary as json")
        f = open(filename, 'w')
        json.dump(self.tag_dict, f)
        f.close()

    def load_tag_dict_as_json(self, filename):
        self.logger.info("loading tag dictionary")
        f = open(filename, 'r')
        self.tag_dict = json.load(f)
        f.close()
        return self.tag_dict

    def load_features(self, feature_path):
        self.logger.info("loading features")
        self.features_mat = np.load(feature_path)

    def create_avg_features_df(self):
        self.logger.info("creating avg features as dataframe")
        tag_dict_avg = {
            tag: np.average(self.features_mat[np.array(ids) - 1], axis=0).tolist()  # average all features for each tag
            for tag, ids in self.tag_dict.items()
        }
        self.feature_avg_df = pd.DataFrame(tag_dict_avg)
        return self.feature_avg_df

    def save_avg_features_df(self, filepath):
        self.logger.info("saving avg features as dataframe")
        self.feature_avg_df.to_pickle(filepath)

    def load_avg_features_df(self, filepath):
        self.logger.info("loading avg features as dataframe")
        self.feature_avg_df = pd.read_pickle(filepath)

