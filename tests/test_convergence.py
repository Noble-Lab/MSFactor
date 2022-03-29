""" tests for convergence criteria in our models """
import sys
import unittest
import pytest
import torch
import numpy as np

sys.path.append('../bin')

from models.linear import GradNMFImputer

class ConvergenceTester(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """ __init__ method for class object """

        # simulated matrix configs
        rng = np.random.default_rng(42) # random seed
        self.matrix_shape = (5,3,3) # (n_rows, n_cols, rank)

        # training params
        self.max_iters = 2000
        self.tol = 1e-4
        self.batch_size = 100
        self.n_factors = 3
        self.learning_rate = 0.1 # default: 1e-3

        # init simulated matrix of known rank 
        W = rng.uniform(size=(self.matrix_shape[0], self.matrix_shape[2]))
        H = rng.uniform(size=(self.matrix_shape[2], self.matrix_shape[1]))
        self.X = W @ H

        # define training set
        self.train = self.X.copy()
        self.train[1,1] = np.nan
        self.train[4,0] = np.nan
        self.train[3,2] = np.nan

        # define validation set
        self.valid = np.empty((self.matrix_shape[0],self.matrix_shape[1]))
        self.valid[:] = np.nan
        self.valid[1,1] = self.X[1,1]
        self.valid[4,0] = self.X[4,0]
        self.valid[3,2] = self.X[3,2]


    def test_basic(self):
        """ do something """
        assert np.linalg.matrix_rank(self.X) == self.matrix_shape[2]

        # init NMF model
        nmf_tester = GradNMFImputer(self.train.shape[0], 
                                    self.train.shape[1], 
                                    n_factors=self.n_factors, 
                                    stopping_tol=self.tol, 
                                    train_batch_size=self.batch_size, 
                                    eval_batch_size=self.batch_size,
                                    n_epochs=self.max_iters, 
                                    optimizer=torch.optim.Adam,
                                    optimizer_kwargs={"lr": self.learning_rate},
                                    )

        # fit & transform
        recon = nmf_tester.fit_transform(self.train, self.valid)

        # get last 10 entries in the model's history attribute
        train_hist_last10 = list(nmf_tester.history["Train MSE"][-10:])
        valid_hist_last10 = list(nmf_tester.history["Validation MSE"][-10:])

        # get MSE means across last 10 entries
        train_mean_last10 = np.mean(train_hist_last10)
        valid_mean_last10 = np.mean(valid_hist_last10)

        # assert each of last 10 entries in history is within a certain
            # absolute tolerance of tol
        for x in train_hist_last10:
            assert np.isclose(x, train_mean_last10, atol=self.tol*10)
            
        for x in valid_hist_last10:
            assert np.isclose(x, valid_mean_last10, atol=self.tol*500)

        # get residuals
        residuals_t = np.abs((train_hist_last10 / train_mean_last10) - 1)
        residuals_v = np.abs((valid_hist_last10 / valid_mean_last10) - 1)

