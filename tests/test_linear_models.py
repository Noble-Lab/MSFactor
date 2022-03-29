""" tests for bin/models/linear.py and bin/models/base.py """
import sys
import unittest
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.append('../bin')
sys.path.append('bin/')

from util_functions import *
from models.linear import GradNMFImputer

class LinearModelsTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """
		self.tol = 1e-4
		self.max_iters = 3000
		self.learning_rate = 0.1 # default 0.01

		#W = np.matrix(np.array([1.0, 3.0, 5.0]))
		#H = np.matrix(np.array([7.0, 11.0]))
		W = np.matrix([1.0, 3.0, 5.0])
		H = np.matrix([7.0, 11.0])

		self.X = np.matmul(W.T, H)

		self.X_missing = self.X.copy()
		self.X_missing[2,1] = np.nan

	def test_lf_1(self):
		""" create a very small matrix with a single missing value; NMF
			imputer should be able to fill in the MV, when trained with lf=1 """

		nmf_tester = GradNMFImputer(self.X.shape[0], self.X.shape[1], 
									n_factors=1, 
									stopping_tol=self.tol, 
									train_batch_size=6, 
									eval_batch_size=6,
									n_epochs=self.max_iters, 
									optimizer=torch.optim.Adam,
                                    optimizer_kwargs={"lr": self.learning_rate})

		# ----------------------------------------------------------------#
		# fully present case
		recon = nmf_tester.fit_transform(self.X)
		
		assert self.X.shape == recon.shape
		assert np.isclose(recon[2,1], 55, atol=1e-5)
		
		row_factors = np.array(nmf_tester.row_factors.weight.detach())
		row_multiplier = 1 / float(row_factors[0])
		row_factors = row_factors * row_multiplier

		print(list(row_factors.flatten()))

		ans1 = np.isclose(list(row_factors.flatten()), [1.0, 3.0, 5.0], atol=1e-4)
		assert ans1.all() # == True

		col_factors = np.array(nmf_tester.col_factors.weight.detach())
		col_multiplier = 7 / float(col_factors[0])
		factors_x = col_factors * col_multiplier

		ans2 = np.isclose(list(factors_x.flatten()), [7.0,11.0], atol=1e-4)
		assert ans2.all() # == True

		# ----------------------------------------------------------------#
		# missing value case
		recon_mv = nmf_tester.fit_transform(self.X_missing)

		assert self.X.shape == recon_mv.shape
		assert np.isclose(recon_mv[2,1], 55, atol=1e-2)

		row_factors = np.array(nmf_tester.row_factors.weight.detach())
		row_multiplier = 1 / float(row_factors[0])
		row_factors = row_factors * row_multiplier

		ans3 = np.isclose(list(row_factors.flatten()), [1.0,3.0,5.0], atol=1e-2)
		assert ans3.all()

		col_factors = np.array(nmf_tester.col_factors.weight.detach())
		col_multiplier = 7 / float(col_factors[0])
		factors_x = col_factors * col_multiplier

		ans4 = np.isclose(factors_x.flatten(), [7.0,11.0], atol=1e-2)
		assert ans4.all()


	def test_lf_2_pv(self):
		""" create a very small matrix with a single missing value; NMF
			imputer should be able to fill in the MV, when trained with lf=2. 
			FULLY PRESENT MATRIX CASE """

		nmf_tester = GradNMFImputer(self.X.shape[0], self.X.shape[1], 
									n_factors=2, 
									stopping_tol=self.tol, 
									train_batch_size=6, 
									eval_batch_size=6,
									n_epochs=self.max_iters,
									optimizer=torch.optim.Adam,
                                    optimizer_kwargs={"lr": self.learning_rate})

		recon = nmf_tester.fit_transform(self.X)
		
		assert self.X.shape == recon.shape
		assert np.isclose(recon[2,1], 55, atol=1e-3)

	@pytest.mark.skip(reason='This one should fail')
	def test_lf_2_mv(self):
		""" same as above, training with lf=2, but this time for 
			MISSING VALUE MATRIX CASE """ 

		nmf_tester = GradNMFImputer(self.X.shape[0], self.X.shape[1], 
									n_factors=2, 
									stopping_tol=self.tol, 
									train_batch_size=6, 
									eval_batch_size=6,
									n_epochs=self.max_iters)


		recon = nmf_tester.fit_transform(self.X_missing)
		
		assert self.X.shape == recon.shape
		assert np.isclose(recon[2,1], 55, atol=1e-3)

