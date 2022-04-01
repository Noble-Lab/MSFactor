"""
TEST-REALISTIC-SIMULATION

This module tests the GradNMFImputer on simulated matrix
that more closely resembles the peptide/protein quant
matrices its designed to work with.
"""
import unittest
import pytest
import os
import pandas as pd 
import numpy as np
import torch
from scipy.stats import ranksums

from ms_imputer.models.linear import GradNMFImputer
import util_functions_test

# simulated matrix configs
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)
long_matrix_shape = (200,10,3)

# training params
n_factors = 3 # for the initial test
tolerance = 0.0001
batch_size = 100
max_iters = 400
learning_rate = 0.05
min_present = 2 # for partition
PXD = "tester"                             

# error assessment params -- these seem like reasonable targets
train_err_tol = 1e-8
test_err_tol = 3.0

class SimulationTesterRealistic(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init the first (basic) simulated matrix
		self.matrix = util_functions_test.simulate_matrix_realistic(
													matrix_shape
		)
		self.train, self.val, self.test = util_functions_test.split(
												self.matrix,
												val_frac=0.1, 
												test_frac=0.1, 
												min_present=min_present
		)
		# init the second (long) simulated matrix
		self.long_matrix = util_functions_test.simulate_matrix_realistic(
													long_matrix_shape
		)
		self.train_long, self.val_long, self.test_long = \
								util_functions_test.split(
										self.long_matrix,
										val_frac=0.1, 
										test_frac=0.1, 
										min_present=min_present
		)

	def test_simulated_realistic(self):
		""" 
		Tests the model's ability to accurately reconstruct a more
		realistic simulated matrix.  
		"""
		model = GradNMFImputer(
						n_rows=self.train.shape[0], 
						n_cols=self.train.shape[1], 
						n_factors=n_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_iters, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate}
		)
		# train
		recon = model.fit_transform(self.train, self.val)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = self.train / 1e18
		val_scaled = self.val / 1e18
		test_scaled = self.test / 1e18
		recon_scaled = recon / 1e18

		train_err = util_functions_test.mse_func_np(train_scaled, recon_scaled)
		val_err = util_functions_test.mse_func_np(val_scaled, recon_scaled)
		test_err = util_functions_test.mse_func_np(test_scaled, recon_scaled)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

	def test_simulated_long(self):
		""" 
		Tests the model's ability to accurately reconstruct a more
		realistic simulated matrix. This time on a tall, skinny matrix  
		"""
		model = GradNMFImputer(
						n_rows=self.train_long.shape[0], 
						n_cols=self.train_long.shape[1], 
						n_factors=n_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_iters, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate}
		)
		# train
		recon = model.fit_transform(self.train_long, self.val_long)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = self.train_long / 1e18
		val_scaled = self.val_long / 1e18
		test_scaled = self.test_long / 1e18
		recon_scaled = recon / 1e18

		train_err = util_functions_test.mse_func_np(train_scaled, recon_scaled)
		val_err = util_functions_test.mse_func_np(val_scaled, recon_scaled)
		test_err = util_functions_test.mse_func_np(test_scaled, recon_scaled)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

	def test_nans_matrix(self):
		"""
		Tests the model's ability to reasonably reconstruct a 
		simulated matrix that has a bunch of np.nans in it
		"""
		# number of indices to withold
		n_nans = int(np.floor(self.matrix.size * 0.4))
		# flatten the initial matrix
		m_flat = self.matrix.flatten()
		# randomly select indices
		rand_idx = np.random.choice(len(m_flat), size=n_nans, replace=False)
		# set to np.nans
		m_flat[rand_idx] = np.nan
		# reshape
		matrix_missing = m_flat.reshape(self.matrix.shape)

		# partition
		train, val, test = util_functions_test.split(
										matrix_missing,
										val_frac=0.1, 
										test_frac=0.1, 
										min_present=1
		)
		# init model
		nmf_model = GradNMFImputer(
						n_rows=train.shape[0], 
						n_cols=train.shape[1], 
						n_factors=n_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_iters, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate}
		)
		# train
		recon = nmf_model.fit_transform(train, val)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = train / 1e18
		val_scaled = val / 1e18
		test_scaled = test / 1e18
		recon_scaled = recon / 1e18

		train_err = util_functions_test.mse_func_np(train_scaled, recon_scaled)
		val_err = util_functions_test.mse_func_np(val_scaled, recon_scaled)
		test_err = util_functions_test.mse_func_np(test_scaled, recon_scaled)

		# check error tolerances
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

