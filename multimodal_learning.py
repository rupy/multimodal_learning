#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import logging
from joint import Joint
import numpy as np

def learn_word2vec_model_from_wiki_corpus():
    joint = Joint()
    joint.learn_wiki_corpus()

# def create_tag_dict():
#     joint = Joint()
#     joint.load_model()
#     joint.create_tag_dict()
#
# def create_feature_matrix():
#     joint = Joint()
#
#     joint.load_model()
#     joint.load_tag_dict()
#     joint.load_flickr_features()
#
#     joint.create_feature_matrices()

def create_features():
    joint = Joint()
    joint.load_model()
    joint.load_flickr_features()
    joint.create_feature_matrices()

def fit_and_transform_by_cca():
    joint = Joint()
    joint.load_feature_matrices()
    joint.fit_data_by_cca()
    joint.transform_data()

def transform_cca():
    joint = Joint()
    joint.load_feature_matrices()
    joint.load_cca()
    joint.transform_data()

def transform_pcca():
    joint = Joint()
    joint.load_feature_matrices()
    joint.load_cca()
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

if __name__== "__main__":

    logging.root.setLevel(level=logging.INFO)
    # fit_and_transform_by_cca()
    # plot_pcca_result()

    create_features()

