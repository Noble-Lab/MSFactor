"""
TEST_INTEGRATION_FAST

Implements some basic integration tests for GradNMFImputer, 
Modeler & Dataset class functionality. Designed to run 
relatively quickly: > 15 sec. 
"""
import sys
import os
import unittest
import pytest
import pandas as pd
import numpy as np
import yaml

sys.path.append('../bin')

from modeler import Modeler, Dataset

# simulated matrix configs
rng = np.random.default_rng(42) # random seed
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)

# training params
n_factors = 4 
PXD = "tester"                             
config_pth = "config-tester.yml"

# error assessment params
train_err_tol = 1e-8
test_err_tol = 1e-1

# open the config file
with open(config_pth) as f:
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

class FastTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init simulated matrix, write to csv
		matrix = simulate_matrix(matrix_shape)
		pd.DataFrame(matrix).to_csv("simulated.csv", index=False)

		# get peptide quants Dataset object
		self.peptides_dataset = Dataset(pxd=PXD, 
										data_pth="./simulated.csv")

		# partition
		self.peptides_dataset.partition(val_frac=0.1, test_frac=0.1, min_present=1)

	def test_integration_fast(self):
		""" 
		Tests GradNMFImputer, Dataset and Modeler class's ability 
		to train an NMF model, and reconstruct the input matrix 
		within a reasonable error tolerance. 
		"""
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
		recon = model.fit(
					self.peptides_dataset.train, 
					self.peptides_dataset.val, 
					n_factors=n_factors, 
					pxd=PXD
				)

		# get errors
		train_err, val_err, test_err = model.collect_errors(
												self.peptides_dataset.train, 
												self.peptides_dataset.val, 
												self.peptides_dataset.test, 
												recon, 
												n_factors
										)
		assert train_err["error"] < train_err_tol
		assert val_err["error"] < test_err_tol
		assert test_err["error"] < test_err_tol

		assert True == True

