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
import ms_imputer.util_functions

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
test_err_tol = 1.0

def simulate_matrix_realistic(matrix_shape):
	"""
	Init a simulated matrix of known size and (approximate) rank. 
	The values of quants_mean and quants_std were derived from a 
	real peptide quants matrix, and should allow us to generate a 
	matrix that more accurately simulates a real peptide quants 
	dataset. Note that taking the abs value of W and H most likely
	changes the true rank of the matrix, thus the assert statement
	in here won't necessarily pass. 

	Parameters
	----------
	matrix_shape: tuple, (x,y,z) where x=n_rows, y=n_cols
	                and z=rank
	Returns
	-------
	X : np.ndarray, the simulated matrix
	"""
	quants_mean = 102161962.5
	quants_std = 978349975.6

	matrix_shape = (12, 10, 3) # (n_rows, n_cols, rank)
	W = np.abs(np.random.normal(loc=quants_mean, scale=quants_std, size=(matrix_shape[0], matrix_shape[2])))
	H = np.abs(np.random.normal(loc=quants_mean, scale=quants_std, size=(matrix_shape[2], matrix_shape[1])))

	X = W @ H

	# won't necessarily pass
	#assert np.linalg.matrix_rank(X) == matrix_shape[2]

	return X

def train_nmf_model(train_mat, valid_mat, lf):
	""" 
	Train a single NMF model, with some given parameters
	
	Parameters
	----------
	train_mat : np.ndarray, the training matrix
	valid_mat : np.ndarray, the validation matrix
	lf : int, number of latent factors to use for reconstruction

	Returns
	-------
	model : GradNMFImputer, the fitted GradNMFImputer object
	recon_mat : np.ndarray, the reconstructed matrix
	"""
	model = GradNMFImputer(
				n_rows=train_mat.shape[0], 
				n_cols=train_mat.shape[1], 
				n_factors=lf, 
				stopping_tol=tolerance, 
				train_batch_size=batch_size, 
				eval_batch_size=batch_size,
				n_epochs=max_iters, 
				loss_func="MSE",
				optimizer=torch.optim.Adam,
				optimizer_kwargs={"lr": learning_rate}
	)

	recon_mat = model.fit_transform(train_mat, valid_mat)

	return model, recon_mat

class SimulationTesterRealistic(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init the first (basic) simulated matrix
		self.matrix = simulate_matrix_realistic(matrix_shape)

		self.train, self.val, self.test = \
						ms_imputer.util_functions.split(
							self.matrix,
							val_frac=0.1, 
							test_frac=0.1, 
							min_present=min_present
		)
		# init the second (long) simulated matrix
		self.long_matrix = simulate_matrix_realistic(long_matrix_shape)

		self.train_long, self.val_long, self.test_long = \
						ms_imputer.util_functions.split(
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
		nmf_model, recon = train_nmf_model(self.train, self.val, n_factors)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = self.train / 1e18
		val_scaled = self.val / 1e18
		test_scaled = self.test / 1e18
		recon_scaled = recon / 1e18

		train_err = ms_imputer.util_functions.mse_func_np(train_scaled, recon_scaled)
		val_err = ms_imputer.util_functions.mse_func_np(val_scaled, recon_scaled)
		test_err = ms_imputer.util_functions.mse_func_np(test_scaled, recon_scaled)

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
		nmf_model, recon = train_nmf_model(self.train_long, self.val_long, n_factors)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = self.train_long / 1e18
		val_scaled = self.val_long / 1e18
		test_scaled = self.test_long / 1e18
		recon_scaled = recon / 1e18

		train_err = ms_imputer.util_functions.mse_func_np(train_scaled, recon_scaled)
		val_err = ms_imputer.util_functions.mse_func_np(val_scaled, recon_scaled)
		test_err = ms_imputer.util_functions.mse_func_np(test_scaled, recon_scaled)

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
		train, val, test = ms_imputer.util_functions.split(
										matrix_missing,
										val_frac=0.1, 
										test_frac=0.1, 
										min_present=1
		)
		# fit model
		nmf_model, recon = train_nmf_model(train, val, lf=n_factors)

		# rescale by a constant, for easier error tolerance calculation
		train_scaled = train / 1e18
		val_scaled = val / 1e18
		test_scaled = test / 1e18
		recon_scaled = recon / 1e18

		train_err = ms_imputer.util_functions.mse_func_np(train_scaled, recon_scaled)
		val_err = ms_imputer.util_functions.mse_func_np(val_scaled, recon_scaled)
		test_err = ms_imputer.util_functions.mse_func_np(test_scaled, recon_scaled)

		# check error tolerances
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

