#!/usr/bin/python
#-*- coding: utf-8 -*-

__author__ = 'rupy'

import numpy as np
from scipy.linalg import eig
import time
import logging
import os
import sys
try:
   import cPickle as pickle
except:
   import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class MyCCA(object):

    def __init__(self, n_components=2, reg_param=0.1, calc_time=False):

        # log setting
        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # CCA params
        self.n_components = n_components
        self.reg_param = reg_param
        self.calc_time = calc_time

        # data
        self.X = None
        self.Y = None

        # Result of fitting
        self.x_weights = None
        self.y_weights = None
        self.eigvals = None
        self.Cxx = None
        self.Cyy = None
        self.Cxy = None

        # transformed data by CCA
        self.X_c = None
        self.Y_c = None

        # transformed data by PCCA
        self.X_pc = None
        self.Y_pc = None
        self.Z_pc = None

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

        self.X = xs
        self.Y = ys

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

        self.Cxx = Cxx
        self.Cyy = Cyy
        self.Cxy = Cxy

        # left = A, right = B
        self.logger.info("solving")
        xleft = np.dot(Cxy, np.linalg.solve(Cyy,Cxy.T))
        xright = Cxx
        x_eigvals, x_eigvecs = self.solve_eigprob(xleft, xright)

        yleft = np.dot(Cxy.T, np.linalg.solve(Cxx,Cxy))
        yright = Cyy
        y_eigvals, y_eigvecs = self.solve_eigprob(yleft, yright)

        self.x_weights = x_eigvecs
        self.eigvals = x_eigvals
        self.y_weights = y_eigvecs

        if self.calc_time:
            print "Fitting done in %.2f sec." % (time.time() - start)

        # print x_eigvals.shape, y_eigvecs.shape
        # print x_eigvals


    def transform(self, x, y, normalize=False):

        if normalize:
            self.logger.info("Normalizing")
            x = self.normalize(x)
            y = self.normalize(y)

        # self.X = x
        # self.Y = y

        self.logger.info("transform matrices by CCA")
        # print x.shape, x_eigvecs[:,:dim].shape
        # print y.shape, y_eigvecs[:,:dim].shape
        x_projected = np.dot(x, self.x_weights[:,:self.n_components])
        y_projected = np.dot(y, self.y_weights[:,:self.n_components])

        self.X_c = x_projected
        self.Y_c = y_projected

        return x_projected, y_projected

    def ptransform(self, x, y, beta=0.5):

        start = time.time()
        print x.shape
        x = self.normalize(x)
        y = self.normalize(y)

        # print x.shape, self.x_weights_.shape
        x_projected = np.dot(x, self.x_weights)
        y_projected = np.dot(y, self.y_weights)

        I = np.eye(len(self.eigvals))
        lamb = np.diag(self.eigvals)
        mat1 = np.linalg.solve(I - np.diag(self.eigvals**2), I)
        mat2 = -np.dot(mat1, lamb)
        mat12 = np.vstack((mat1, mat2))
        mat21 = np.vstack((mat2, mat1))
        mat = np.hstack((mat12, mat21))
        # print lamb.shape, lamb
        p = np.vstack((lamb**beta, lamb**(1-beta)))
        q = np.vstack((x_projected.T, y_projected.T))
        print p.T.shape, mat.shape, q.shape
        z = np.dot(p.T, np.dot(mat, q)).T[:,:self.n_components]

        if self.calc_time:
            print "Transforming done in %.2f sec." % (time.time() - start)

        self.X_pc = x_projected
        self.Y_pc = y_projected
        self.Z_pc = z

        return x_projected, y_projected, z

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

    def fit_ptransform(self, x, y, beta=0.5):
        self.fit(x, y)
        return self.ptransform(x, y, beta)

    def save_params_as_pickle(self, filename):
        data = (self.n_components, self.reg_param, self.x_weights , self.y_weights, self.eigvals, self.calc_time,
                self.X, self.Y, self.Cxx, self.Cyy, self.Cxy)
        self.logger.info("saving cca")
        f = open(filename, 'wb')
        pickle.dump(data, f)
        f.close()

    def load_params_from_pickle(self, filename):
        self.logger.info("loading cca")
        f = open(filename, 'rb')
        self.n_components, self.reg_param, self.x_weights , self.y_weights, self.eigvals, self.calc_time,\
        self.X, self.Y, self.Cxx, self.Cyy, self.Cxy = pickle.load(f)
        f.close()

    def check_fit_finished(self):
        return self.x_weights is not None and self.y_weights is not None and self.eigvals is not None and self.X is not None and self.Y is not None and self.Cxx is not None and self.Cyy is not None and self.Cxy is not None

    def plot_cca_result(self, probabilistic=False):

        X = None
        Y = None
        Z = None
        if probabilistic:
            self.logger.info("plotting PCCA")
            X = self.X_pc
            Y = self.Y_pc
            Z = self.Z_pc
        else:
            self.logger.info("plotting CCA")
            X = self.X_c
            Y = self.Y_c


        # PCA
        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)
        Y_r = pca.fit(Y).transform(Y)
        Z_r = None
        if probabilistic:
            Z_r = pca.fit(Z).transform(Z)

        # begin plot
        plt.figure()

        plt.subplot(221)
        plt.plot(Y_r[:, 0], Y_r[:, 1], 'xb')
        plt.plot(X_r[:, 0], X_r[:, 1], '.r')
        plt.title('CCA XY')

        plt.subplot(222)
        plt.plot(X_r[:, 0], X_r[:, 1], '.r')
        plt.title('CCA X')

        plt.subplot(223)
        plt.plot(Y_r[:, 0], Y_r[:, 1], 'xb')
        plt.title('CCA Y')

        if probabilistic:
            plt.subplot(224)
            plt.plot(Z_r[:, 0], Z_r[:, 1], 'xb')
            plt.title('CCA Z')

        plt.show()



    def normalize(self, mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        # for arr in mat:
        #     m = np.mean(arr)
        #     # s = np.std(arr)
        #     # if s != 0.0:
        #     arr = np.array([(arr[i] - m) for i in range(len(arr))])
        return mat

    def corrcoef(self):
        return np.corrcoef(self.X_c[:,0], self.Y_c[:,0])

if __name__=="__main__":

    # Reduce dimensions of x, y from 30, 20 to 10 respectively.
    x = np.random.random((100, 30))
    y = np.random.random((100, 20))
    cca = MyCCA(n_components=10, reg_param=0.1, calc_time=True)
    x_c, y_c = cca.fit_transform(x, y)

    #
    print np.corrcoef(x_c[:,0], y_c[:,0])