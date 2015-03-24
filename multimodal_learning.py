#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
from joint import Joint
import numpy as np

def learn_word2vec_model_from_wiki_corpus():
    joint = Joint()
    joint.learn_wiki_corpus()

def create_tag_list():
    joint = Joint()
    joint.create_tag_list()

def create_word_feature():
    joint = Joint()
    joint.create_word_feature_matrix()

def create_image_feature():
    joint = Joint()
    joint.create_image_feature_matrix()

def fit_by_cca():
    joint = Joint()
    joint.fit_data_by_cca()

def transform_cca():
    joint = Joint()
    joint.transform_data()

def transform_pcca():
    joint = Joint()
    joint.transform_data(probabilistic=True)

def print_corrcoef():
    joint = Joint()
    joint.load_transformed_data()
    joint.print_corrcoef()

def plot_cca_result():
    joint = Joint()
    joint.load_transformed_data()
    joint.plot_transformed_data()

def plot_pcca_result():
    joint = Joint()
    joint.load_transformed_data(probabilistic=True)
    joint.plot_transformed_data(probabilistic=True)

def create_labels():
    joint = Joint()
    joint.create_labels()

def plot_tag_data(search_tag):
    joint = Joint()
    joint.load_transformed_data()
    joint.load_labels()
    joint.plot_tag_data(search_tag)

def plot_tag_img_pairs(search_tag):
    joint = Joint()
    joint.load_transformed_data()
    joint.load_labels()
    joint.plot_tag_img_pairs(search_tag, 30)

def nn_by_tag(search_tag):
    joint = Joint()
    joint.load_transformed_data()
    joint.pca_fit(30)
    joint.load_labels()
    joint.tag_nearest_neighbor(search_tag, 30)

def plot_img_data(dataset_idx):
    joint = Joint()
    joint.load_transformed_data()
    joint.load_labels()
    joint.plot_img_data(dataset_idx)

def nn_by_img(dataset_id):
    joint = Joint()
    joint.load_transformed_data()
    joint.pca_fit(30)
    joint.load_labels()
    joint.img_nearest_neighbor(dataset_id, 30)

def plot_images_by_tag(search_tag):
    joint = Joint()
    joint.plot_img_by_tag(search_tag)

def plot_image_in_plot(img_id):
    joint = Joint()
    joint.plot_img_in_plot(img_id)


if __name__== "__main__":

    logging.root.setLevel(level=logging.INFO)
    # fit_and_transform_by_cca()
    # plot_pcca_result()

    # transform_pcca()
    # transform_cca()
    # create_labels()

    # word = "dog"
    # plot_tag_img_pairs("horse")
    # plot_tag_data(word)
    # nn_by_tag(word)
    # plot_images_by_tag("cow")
    # plot_cca_result()

    id = 164
    plot_img_data(id)
    nn_by_img(id)

    # plot_image_in_plot(1)

