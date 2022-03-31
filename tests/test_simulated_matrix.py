"""
TEST-SIMULATED-MATRIX

The SimulationTester class implements a couple of tests 
designed to assess basic funcitonality of the 
GradNMFImputer class, among others. Tests are performed
on a small simulated matrix. 
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
rng = np.random.default_rng(42) # random seed
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)
long_matrix_shape = (200,8,2)

# training params
n_factors = 4 # for the first test
tolerance = 0.0001
batch_size = 100
max_iters = 400
learning_rate = 0.05
min_present = 2 # for partition
PXD = "tester"                             

# error assessment params
train_err_tol = 1e-8
test_err_tol = 1e-1

def simulate_matrix(matrix_shape):
	""" 
	Init simulated matrix of known size and rank

	Parameters
	----------
	matrix_shape: tuple, (x,y,z) where x=n_rows, y=n_cols
				  and z=rank
	Returns
	-------
	X: np.ndarray, the simulated matrix
	"""
	W = rng.uniform(size=(matrix_shape[0], matrix_shape[2]))
	H = rng.uniform(size=(matrix_shape[2], matrix_shape[1]))
	X = W @ H

	assert np.linalg.matrix_rank(X) == matrix_shape[2]

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

class SimulationTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init the first (basic) simulated matrix
		self.matrix = simulate_matrix(matrix_shape)
		pd.DataFrame(self.matrix).to_csv("simulated.csv", index=False)

		self.train, self.val, self.test = \
						ms_imputer.util_functions.split(
							self.matrix,
							val_frac=0.1, 
							test_frac=0.1, 
							min_present=min_present
		)
		# init the second (long) simulated matrix
		self.long_matrix = simulate_matrix(long_matrix_shape)
		pd.DataFrame(self.long_matrix).to_csv("simulated.csv", index=False)

		self.train_long, self.val_long, self.test_long = \
						ms_imputer.util_functions.split(
							self.long_matrix,
							val_frac=0.1, 
							test_frac=0.1, 
							min_present=min_present
		)
		os.remove("simulated.csv")

	def test_split(self):
		"""
		Make sure the split function is doing roughly
		what its supposed to
		"""
		train_nans = np.count_nonzero(np.isnan(self.train))
		val_nans = np.count_nonzero(np.isnan(self.val))
		test_nans = np.count_nonzero(np.isnan(self.test))

		train_frac = (self.train.size - train_nans) / self.train.size
		val_frac = (self.val.size - val_nans) / self.val.size
		test_frac = (self.test.size - test_nans) / self.test.size

		assert 0.9 > train_frac > 0.7
		assert 0.25 > val_frac > 0.0
		assert 0.25 > test_frac > 0.0

		assert np.isclose(val_frac, test_frac, atol=0.1)

		assert self.train.shape == self.val.shape == self.test.shape 
		assert self.train.shape == self.matrix.shape

		assert np.array_equal(self.matrix, self.train) == False
		assert np.array_equal(self.matrix, self.val) == False
		assert np.array_equal(self.matrix, self.test) == False

		# make sure number of nans per row is less than min_present
		nans_x_row = np.count_nonzero(np.isnan(self.train), axis=1)
		assert nans_x_row.max() < self.matrix.shape[1] - min_present

	def test_nmf_imputer_simulated(self):
		""" 
		Tests GradNMFImputer class's ability to train an 
		NMF model, and reconstruct a small simulated matrix 
		within a reasonable error tolerance. 
		"""
		nmf_model, recon = train_nmf_model(self.train, self.val, lf=1)

		train_err = ms_imputer.util_functions.mse_func_np(self.train, recon)
		val_err = ms_imputer.util_functions.mse_func_np(self.val, recon)
		test_err = ms_imputer.util_functions.mse_func_np(self.test, recon)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

		# make sure shape of reconstructed matrix is correct
		assert recon.shape == self.train.shape

		# make sure model isn't imputing a bunch of zeros
		imputed_zero_cts = np.count_nonzero(np.isnan(recon))
		assert imputed_zero_cts < self.train.size * 0.1

		# make sure model isn't imputing extreme values
		assert recon.max() < self.matrix.max() * 2
		assert recon.min() > self.matrix.min() / 2

		# make sure all predictions are positive
		assert np.count_nonzero(recon < 0) == 0

		# make sure recon matrix isn't just the input matrix 
		assert np.array_equal(self.matrix, recon) == False

		n_epochs = nmf_model.history.shape[0] - 1

		# make sure that validation loss has decreased
		if n_epochs > 15:
			# validation loss
			window2 = np.array(nmf_model.history["Validation MSE"][0:15])
			window1 = np.array(nmf_model.history["Validation MSE"][-15:])

			val_wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

			assert val_wilcoxon_p < 0.05

			# training loss
			window2 = np.array(nmf_model.history["Train MSE"][0:15])
			window1 = np.array(nmf_model.history["Train MSE"][-15:])

			train_wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

			assert train_wilcoxon_p < 0.05

	def test_overdetermined(self):
		""" 
		Test NMF model in an overdetermined setting, where we provide 
		way more latent factors than is optimal. Model should still arrive
		at a reasonable solution
		"""
		nmf_model, recon = train_nmf_model(self.train, self.val, lf=12)

		train_err = ms_imputer.util_functions.mse_func_np(self.train, recon)
		val_err = ms_imputer.util_functions.mse_func_np(self.val, recon)
		test_err = ms_imputer.util_functions.mse_func_np(self.test, recon)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable...multiplying error tols by a constant
		assert train_err < train_err_tol
		assert val_err < test_err_tol * 4
		assert test_err < test_err_tol * 4

	def test_underdetermined(self):
		""" 
		Test NMF model in an underdetermined setting, where we provide 
		fewer latent factors than is optimal. Model should still arrive
		at a reasonable solution
		"""
		nmf_model, recon = train_nmf_model(self.train, self.val, lf=1)

		train_err = ms_imputer.util_functions.mse_func_np(self.train, recon)
		val_err = ms_imputer.util_functions.mse_func_np(self.val, recon)
		test_err = ms_imputer.util_functions.mse_func_np(self.test, recon)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

	def test_long_matrix(self):
		"""
		Test an NMF model trained on a (simulated) long skinny matrix. 
		This should more closely resemble the peptide quants matrices
		our model is designed to work with. 
		"""
		nmf_model, recon = train_nmf_model(self.train_long, self.val_long, lf=2)

		train_err = ms_imputer.util_functions.mse_func_np(self.train_long, recon)
		val_err = ms_imputer.util_functions.mse_func_np(self.val_long, recon)
		test_err = ms_imputer.util_functions.mse_func_np(self.test_long, recon)

		# make sure error tolerances of predictions for all 
		#    three sets are reasonable
		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

		# make sure model isn't imputing a bunch of zeros
		imputed_zero_cts = np.count_nonzero(np.isnan(recon))
		assert imputed_zero_cts < self.train.size * 0.1

