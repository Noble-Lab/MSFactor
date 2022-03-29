""" 
TEST_OOP

Testing the newly refactored object oriented fit_models code. 
Comparing the reconstructions I get with the new object oriented 
code versus calling the GradNMFImputer class directly, 
which is essentially what happens within the old fit_models.py script. 
For xx bootstrap iterations, where I generate a single simulated
matrix, and repartition into train, validation & test for each 
iteration.

This test is very slow!
"""
import sys
import os
import unittest
import pytest
import pandas as pd
import numpy as np
import torch
import yaml
from scipy.stats import ranksums

sys.path.append('../bin')

from modeler import Modeler
from models.linear import GradNMFImputer
from util_functions import mse_func_np, split

# simulated matrix configs
rng = np.random.default_rng(42) # random seed
MATRIX_SHAPE = (100,40,4) # (n_rows, n_cols, rank)

# for error assessment
TRAIN_TOL = 1e-8
VAL_TOL = 1e-1
ALPHA = 0.1 # alpha to use for lack of significance test

BOOTSTRAP_ITERS = 5  # increasing this would yield more robust 
					 #    results, but would run slower
# training params
N_FACTORS = 4 
PXD = "tester"                             
CONFIG_PTH = "config-tester.yml"

# open the config file
with open(CONFIG_PTH) as f:
	configs = yaml.load(f, Loader=yaml.FullLoader)

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


def get_close_fractions(recon, train, val, test, train_tol, val_tol):
	""" 
	Calculates the fraction of reconstructed train, validation and test set
	values that are within some tolerance of the ground truth

	Parameters
	----------
	train, val, test : np.ndarrays, the training, validation and test sets
	train_tol, val_tol: The allowable error tolerance, training set and validation 
						set, to still be considered "close"
	Returns
	-------
	train_close_frac, val_close_frac, test_close_frac : The fraction of
		reconstructed train, validation and test set values that are within some
		tolerance of the ground truth. 
	"""
	# get masks
	train_mask = np.isnan(train)
	val_mask = np.isnan(val)
	test_mask = np.isnan(test)

	# how many reconstructed values are within some tolerance of the ground truth?
	train_close = np.isclose(train[~train_mask], recon[~train_mask], atol=train_tol)
	val_close = np.isclose(val[~val_mask], recon[~val_mask], atol=val_tol)
	test_close = np.isclose(test[~test_mask], recon[~test_mask], atol=val_tol)

	# get percentages
	train_close_frac = np.count_nonzero(train_close) / len(train[~train_mask])
	val_close_frac = np.count_nonzero(val_close) / len(val[~val_mask])
	test_close_frac = np.count_nonzero(test_close) / len(test[~test_mask])

	return train_close_frac, val_close_frac, test_close_frac


def direct_modeler(train, val, test):
	""" 
	Train a NMF model by directly calling the GradNMFImputer class...this
	is essentially what the old fit_models.py workflow was doing

	Parameters
	----------
	train, val, test : np.ndarray, the training, validation and test sets

	Returns
	-------
	train_err, val_err, test_err : float, reconstruction error for each of
									the training, validation and test sets
	"""
	nmf_model = GradNMFImputer(train.shape[0], 
								train.shape[1], 
								n_factors=N_FACTORS, 
								stopping_tol=configs['tolerance'], 
								train_batch_size=configs['batch_size'], 
								eval_batch_size=configs['batch_size'],
								n_epochs=configs['max_iters'], 
								loss_func=configs['loss_func'],
								patience=configs['patience'],
								optimizer=torch.optim.Adam,
								optimizer_kwargs={"lr": configs['learning_rate']})

	# fit and transform
	recon = nmf_model.fit_transform(train, val)

	# how close are reconstructed values to the ground truth? 
	train_close, val_close, test_close = get_close_fractions(
											recon, train, val, test, 
											TRAIN_TOL, VAL_TOL
										)

	train_loss = mse_func_np(train, recon)
	valid_loss = mse_func_np(val, recon)
	test_loss = mse_func_np(test, recon)

	return train_loss, valid_loss, test_loss


def object_oriented_modeler(train, val, test):
	""" 
	Train a NMF model using our new object oriented code, and return the
	reconstruction error. 

	Parameters
	----------
	train, val, test : np.ndarray, the training, validation and test sets

	Returns
	-------
	train_err, val_err, test_err : float, reconstruction error for each of
									the training, validation and test sets
	"""
	# init model 
	model = Modeler(
				model_type="NMF", 
				loss_func=configs["loss_func"],
				log_transform=configs["log_transform"],
				tolerance=configs["tolerance"],
				max_epochs=configs["max_iters"],
				batch_size=configs["batch_size"],
				learning_rate=configs["learning_rate"],
				parallelize=configs["parallelize"],
		)

	# fit model, get reconstruction
	recon = model.fit(train, val, n_factors=N_FACTORS, pxd=PXD)

	# how close are reconstructed values to the ground truth? 
	train_close, val_close, test_close = get_close_fractions(
											recon, train, val, test, 
											TRAIN_TOL, VAL_TOL
										)
	# get errors
	train_err, val_err, test_err = model.collect_errors(train, 
														val, 
														test, 
														recon, 
														N_FACTORS)
	return train_err, val_err, test_err


class ObjectOrientedTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init simulated matrix, write to csv
		self.X = simulate_matrix(MATRIX_SHAPE)
		pd.DataFrame(self.X).to_csv("simulated.csv", index=False)

		# init list to store results
		self.results_sim = []
		self.results_quants = []

		# read in real (small) peptide quants dataset
		peptides_pth = "../data/peptides-data/PXD014156_peptides.csv"
		self.peptide_quants = pd.read_csv(peptides_pth)
		self.peptide_quants.replace(0, np.nan, inplace=True)
		self.peptide_quants = self.peptide_quants.to_numpy()


	def test_bootstrap_simulated(self):
		""" 
		run bootstrap iterations, record errors for each partition. 
		data are partitioned before each iteration. For a simulated
		dataset
		"""
		for n_iter in range(0,BOOTSTRAP_ITERS):
			# partition
			train, val, test = split(self.X, 
										val_frac=configs['valid_frac'],
										test_frac=configs['test_frac'], 
										min_present=configs['min_present']
									)
			# run object oriented code
			train_err_oo, val_err_oo, test_err_oo = object_oriented_modeler(
														train, val, test,
													)

			# run direct code
			train_err_d, val_err_d, test_err_d = direct_modeler(train, val, test)

			res = {
				"iter": n_iter,
				"train_error_oo": train_err_oo["error"],
				"valid_error_oo": val_err_oo["error"],
				"test_error_oo": test_err_oo["error"],
				"train_error_d": train_err_d,
				"valid_error_d": val_err_d,
				"test_error_d": test_err_d,
			}
		    
			self.results_sim += [pd.DataFrame(res, index=[0])]

		# format results
		self.results_sim = pd.concat(self.results_sim)
		self.results_sim = self.results_sim.reset_index(drop=True)

		# collect wilcoxon rank sum p values
		train_wilcoxon_p = ranksums(self.results_sim["train_error_oo"], 
									self.results_sim["train_error_d"], 
									alternative="two-sided")[1]
		valid_wilcoxon_p = ranksums(self.results_sim["valid_error_oo"], 
									self.results_sim["valid_error_d"], 
									alternative="two-sided")[1]
		test_wilcoxon_p = ranksums(self.results_sim["test_error_oo"], 
									self.results_sim["test_error_d"], 
									alternative="two-sided")[1]

		# checking for LACK OF SIGNIFIGANCE, with alpha = ALPHA
		assert train_wilcoxon_p > ALPHA
		assert valid_wilcoxon_p > ALPHA
		assert test_wilcoxon_p > ALPHA

		assert True == True


	def test_bootstrap_peptides(self):
		"""
		run bootstrap iterations, record errors for each partition. 
		data are partitioned before each iteration. For a real peptide
		quants dataset this time. 
		"""
		for n_iter in range(0,BOOTSTRAP_ITERS):
			# partition
			train, val, test = split(self.peptide_quants, 
									val_frac=configs['valid_frac'],
									test_frac=configs['test_frac'], 
									min_present=configs['min_present']
									)
			# run object oriented code
			train_err_oo, val_err_oo, test_err_oo = object_oriented_modeler(
														train, val, test,
													)

			# run direct code
			train_err_d, val_err_d, test_err_d = direct_modeler(train, val, test)

			res = {
				"iter": n_iter,
				"train_error_oo": train_err_oo["error"],
				"valid_error_oo": val_err_oo["error"],
				"test_error_oo": test_err_oo["error"],
				"train_error_d": train_err_d,
				"valid_error_d": val_err_d,
				"test_error_d": test_err_d,
			}

			self.results_quants += [pd.DataFrame(res, index=[0])]

		# format results
		self.results_quants = pd.concat(self.results_quants)
		self.results_quants = self.results_quants.reset_index(drop=True)

		# collect wilcoxon rank sum p values
		train_wilcoxon_p = ranksums(self.results_quants["train_error_oo"], 
									self.results_quants["train_error_d"], 
									alternative="two-sided")[1]
		valid_wilcoxon_p = ranksums(self.results_quants["valid_error_oo"], 
									self.results_quants["valid_error_d"], 
									alternative="two-sided")[1]
		test_wilcoxon_p = ranksums(self.results_quants["test_error_oo"], 
									self.results_quants["test_error_d"], 
									alternative="two-sided")[1]

		# checking for LACK OF SIGNIFIGANCE, with alpha = ALPHA
		assert train_wilcoxon_p > ALPHA
		assert valid_wilcoxon_p > ALPHA
		assert test_wilcoxon_p > ALPHA

		assert True == True

