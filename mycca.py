#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
from scipy.linalg import eig
import time
import logging
import os
import sys

class MyCCA(object):

    def __init__(self, n_components=2, reg_param=0.1, calc_time=False):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        self.n_components = n_components
        self.reg_param = reg_param
        self.x_weights_ = None
        self.x_eigvals_ = None
        self.y_weights_ = None
        self.y_eigvals_ = None
        self.calc_time = calc_time

    def get_params(self):
        print "===================="
        print "  CCA parameters    "
        print "===================="
        print " | "
        print " |- n_components: %s" % self.n_components
        print " |- reg_param:    %s" % self.reg_param
        print " |- calc_time:    %s" % self.calc_time

    def solve_eigprob(self, left, right):

        self.logger.info("calculating eigen dimension")
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])
        # print eig_dim

        self.logger.info("calculating eigenvalues and eigenvector")
        eig_vals, eig_vecs = eig(left, right)# ;print eig_vals.imag

        self.logger.info("sorting eigenvalues and eigenvector")
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        # regularization
        self.logger.info("regularizing")
        eig_vecs = np.dot(eig_vecs, np.diag(np.reciprocal(np.linalg.norm(eig_vecs, axis=0))))

        return eig_vals, eig_vecs


    def fit(self, xs, ys):

        self.logger.info("calculating average, variance, and covariance")
        z = np.vstack((xs.T, ys.T))
        Cov = np.cov(z)
        p = len(xs.T)
        Cxx = Cov[:p, :p]
        Cyy = Cov[p:, p:]
        Cxy = Cov[:p, p:]
        # print Cxx.shape, Cxy.shape, Cyy.shape


        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")

        start = time.time()

        self.logger.info("adding regularization term")
        Cxx += self.reg_param * np.average(np.diag(Cxx)) * np.eye(Cxx.shape[0])
        Cyy += self.reg_param * np.average(np.diag(Cyy)) * np.eye(Cyy.shape[0])

        # left = A, right = B
        self.logger.info("solving")
        xleft = np.dot(Cxy, np.linalg.solve(Cyy,Cxy.T))
        xright = Cxx
        x_eigvals, x_eigvecs = self.solve_eigprob(xleft, xright)

        yleft = np.dot(Cxy.T, np.linalg.solve(Cxx,Cxy))
        yright = Cyy
        y_eigvals, y_eigvecs = self.solve_eigprob(yleft, yright)

        self.x_weights_ = x_eigvecs
        self.x_eigvals_ = x_eigvals
        self.y_weights_ = y_eigvecs
        self.y_eigvals_ = y_eigvals

        if self.calc_time:
            print "Fitting done in %.2f sec." % (time.time() - start)

        # print x_eigvals.shape, y_eigvecs.shape
        # print x_eigvals


    def transform(self, x, y):
        self.logger.info("transform matrices by CCA")
        # print x.shape, x_eigvecs[:,:dim].shape
        # print y.shape, y_eigvecs[:,:dim].shape
        x_projected = np.dot(x, self.x_weights_[:,:self.n_components])
        y_projected = np.dot(y, self.y_weights_[:,:self.n_components])

        return x_projected, y_projected


    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)



if __name__=="__main__":

    # Reduce dimensions of x, y from 30, 20 to 10 respectively.
    x = np.random.random((100, 30))
    y = np.random.random((100, 20))
    cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)
    x_c, y_c = cca.fit_transform(x, y)

    #
    print np.corrcoef(x_c[:,0], y_c[:,0])