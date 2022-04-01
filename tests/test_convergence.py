"""
TEST-CONVERGENCE

This module tests the convergence criteria of the
GradNMFImputer class. Also tests GradNMFImputer on 
a very small example of a real peptide quants matrix, 
peptide_quants_tester.csv. Arguably these should be
separate tests.
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

# training params
n_factors = 3 # for the initial test
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)
tolerance = 0.0001
batch_size = 100
max_iters = 400
learning_rate = 0.05
min_present = 2 # for partition
lf = 4
train_err_tol = 1e-1
test_err_tol = 1e15 # needs to be super high, for real peptide quants 

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

def train_nmf_model(train_mat, valid_mat, lf, _tol):
	""" 
	Train a single NMF model, with some given parameters
	
	Parameters
	----------
	train_mat : np.ndarray, the training matrix
	valid_mat : np.ndarray, the validation matrix
	lf : int, number of latent factors to use for reconstruction
	tolerance : float, the early stopping error tolerance

	Returns
	-------
	model : GradNMFImputer, the fitted GradNMFImputer object
	recon_mat : np.ndarray, the reconstructed matrix
	"""
	model = GradNMFImputer(
				n_rows=train_mat.shape[0], 
				n_cols=train_mat.shape[1], 
				n_factors=lf, 
				stopping_tol=_tol, 
				train_batch_size=batch_size, 
				eval_batch_size=batch_size,
				n_epochs=max_iters, 
				loss_func="MSE",
				optimizer=torch.optim.Adam,
				optimizer_kwargs={"lr": learning_rate}
	)

	recon_mat = model.fit_transform(train_mat, valid_mat)

	return model, recon_mat


class ConvergenceTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """
		# real matrix
		quants_matrix = pd.read_csv("peptide_quants_tester.csv")

		self.train, self.val, self.test = \
						ms_imputer.util_functions.split(
							quants_matrix,
							val_frac=0.1, 
							test_frac=0.1, 
							min_present=min_present
		)

		self.nmf_model, self.recon = train_nmf_model(
										self.train, 
										self.val, 
										n_factors, 
										_tol = tolerance)

		# simulated matrix
		sim_matrix = simulate_matrix_realistic(matrix_shape)
		self.train_sim, self.val_sim, self.test_sim = \
						ms_imputer.util_functions.split(
									sim_matrix,
									val_frac=0.1, 
									test_frac=0.1, 
									min_present=min_present
		)
		# training without tolerance -> should run out to max_iters
		self.sim_model, self.sim_recon = train_nmf_model(
											self.train_sim, 
											self.val_sim, 
											n_factors, 
											_tol = 0.0)

	def test_reconstruction_real_matrix(self):
		""" 
		Evaluates reconstruction error for a real, but very small,
		peptide-quants matrix (peptide_quants_tester.csv)
		"""
		
		# get the model reconstruction errors
		train_err = ms_imputer.util_functions.mse_func_np(self.train, self.recon)
		val_err = ms_imputer.util_functions.mse_func_np(self.val, self.recon)
		test_err = ms_imputer.util_functions.mse_func_np(self.test, self.recon)

		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

	def test_convergence(self):
		"""
		Test convergence criteria for the GradNMFImputer class
		"""
		if self.nmf_model.early_stopping == "wilcoxon": 
			window2 = np.array(self.nmf_model.history["Validation MSE"][-15:])
			window1 = np.array(self.nmf_model.history["Validation MSE"][-35:-20])

			wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

			assert wilcoxon_p < 0.05
    
		elif self.nmf_model.early_stopping == "standard":
			stopping_counter = 0
			best_loss = np.min(self.nmf_model.history["Validation MSE"])

			for val_loss in self.nmf_model.history["Validation MSE"][-10:]:
				tol = np.abs((best_loss - val_loss) / best_loss)
				loss_ratio = val_loss / best_loss

				if tol < tolerance:
					stopping_counter += 1
				else:
					stopping_counter = 0
		
			# assuming that patience == 10
			assert stopping_counter == 10

		else: # early stopping was not triggered
			assert len(self.nmf_model.history.epoch) == max_iters + 1

	def test_convergence_runout(self):
		"""
		Test convergence criteria for GradNMFImputer class. 
		The tolerance was turned off for this one...model should
		have run out to max_iters
		"""
		assert len(self.sim_model.history.epoch) == max_iters + 1
		assert self.sim_model.early_stopping == None

		# but loss should still have decreased
		window2 = np.array(self.sim_model.history["Validation MSE"][0:15])
		window1 = np.array(self.sim_model.history["Validation MSE"][-15:])

		val_wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

		assert val_wilcoxon_p < 0.05

		# training loss
		window2 = np.array(self.sim_model.history["Train MSE"][0:15])
		window1 = np.array(self.sim_model.history["Train MSE"][-15:])

		train_wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

		assert train_wilcoxon_p < 0.05










