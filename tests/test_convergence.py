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
import util_functions_test

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

class ConvergenceTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """
		# REAL MATRIX
		quants_matrix = pd.read_csv("peptide_quants_tester.csv")
		# partition
		self.train, self.val, self.test = \
						util_functions_test.split(
								quants_matrix,
								val_frac=0.1, 
								test_frac=0.1, 
								min_present=min_present
		)
		# init model
		self.real_model = GradNMFImputer(
						n_rows=self.train.shape[0], 
						n_cols=self.train.shape[1], 
						n_factors=lf, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_iters, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate}
		)
		# train
		self.real_recon = self.real_model.fit_transform(self.train, self.val)

		# SIMULATED MATRIX
		sim_matrix = util_functions_test.simulate_matrix_realistic(matrix_shape)
		# partition
		self.train_sim, self.val_sim, self.test_sim = \
						util_functions_test.split(
									sim_matrix,
									val_frac=0.1, 
									test_frac=0.1, 
									min_present=min_present
		)
		# training without tolerance -> should run out to max_iters
		self.sim_model = GradNMFImputer(
						n_rows=self.train_sim.shape[0], 
						n_cols=self.train_sim.shape[1], 
						n_factors=lf, 
						stopping_tol=0.0, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_iters, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate}
		)

		self.sim_recon = self.sim_model.fit_transform(self.train_sim, self.val_sim)

	def test_reconstruction_real_matrix(self):
		""" 
		Evaluates reconstruction error for a real, but very small,
		peptide-quants matrix (peptide_quants_tester.csv)
		"""
		# get the model reconstruction errors
		train_err = util_functions_test.mse_func_np(self.train, self.real_recon)
		val_err = util_functions_test.mse_func_np(self.val, self.real_recon)
		test_err = util_functions_test.mse_func_np(self.test, self.real_recon)

		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

	def test_convergence(self):
		"""
		Test convergence criteria for the GradNMFImputer class
		"""
		if self.real_model.early_stopping == "wilcoxon": 
			window2 = np.array(self.real_model.history["Validation MSE"][-15:])
			window1 = np.array(self.real_model.history["Validation MSE"][-35:-20])

			wilcoxon_p = ranksums(window2, window1, alternative="greater")[1]

			assert wilcoxon_p < 0.05
    
		elif self.real_model.early_stopping == "standard":
			stopping_counter = 0
			best_loss = np.min(self.real_model.history["Validation MSE"])

			for val_loss in self.real_model.history["Validation MSE"][-11:-1]:
				tol = np.abs((best_loss - val_loss) / best_loss)
				loss_ratio = val_loss / best_loss

				if tol < tolerance:
					stopping_counter += 1
				else:
					stopping_counter = 0
		
			# assuming that patience == 10
			assert stopping_counter == 10

		else: # early stopping was not triggered
			assert len(self.real_model.history.epoch) == max_iters + 1

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

