"""
TEST LOSS FUNCTIONS

Basic integration tests for the model's
ability to train, then reconstruct the input 
matrix within a reasonable error tolerance, for
various loss functions:
	MSE,
	relative MSE, 
	Poisson loss, 
	variance corrected MSE
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
log_transform = False
tolerance = 0.001
max_epochs = 3000
batch_size = 100
learning_rate = 0.1

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

def train_model(loss, dataset):
	""" 
	Train a model given the specified loss function, then 
	collect training, validation and test set errors
	
	Parameters
	----------
	loss : str, the loss function to use for
			model training
	dataset : Dataset(), the partitioned peptides dataset
	
	Returns
	-------
	train_err, val_err, test_err : float, the reconstruction
			errors for the training, validation and test sets
	"""
	model = Modeler(
		model_type="NMF", 
		loss_func=loss,
		log_transform=log_transform,
		tolerance=tolerance,
		max_epochs=max_epochs,
		batch_size=batch_size,
		learning_rate=learning_rate,
		parallelize="forests",
	)

	# fit model, get reconstruction
	recon = model.fit(
					dataset.train, 
					dataset.val, 
					n_factors=n_factors, 
					pxd=PXD
			)

	# get errors
	train_err, val_err, test_err = model.collect_errors(
												dataset.train, 
												dataset.val, 
												dataset.test, 
												recon, 
												n_factors
									)

	return train_err, val_err, test_err

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

	def test_mse(self):
		""" 
		Tests the model's ability to train, then reconstruct the input 
		matrix within a reasonable error tolerance. 

		For standard MSE loss function
		"""
		train_loss, val_loss, test_loss = train_model(
											loss="MSE", 
											dataset=self.peptides_dataset
										)


		assert train_loss["error"] < train_err_tol
		assert val_loss["error"] < test_err_tol
		assert test_loss["error"] < test_err_tol

	def test_rmse(self):
		""" 
		Tests the model's ability to train, then reconstruct the input 
		matrix within a reasonable error tolerance. 

		For relative MSE loss function
		"""
		train_loss, val_loss, test_loss = train_model(
											loss="RMSE", 
											dataset=self.peptides_dataset
										)

		assert train_loss["error"] < train_err_tol
		assert val_loss["error"] < test_err_tol
		assert test_loss["error"] < test_err_tol

	def test_poisson(self):
		""" 
		Tests the model's ability to train, then reconstruct the input 
		matrix within a reasonable error tolerance. 

		For Poisson loss function
		"""
		train_loss, val_loss, test_loss = train_model(
											loss="Poisson", 
											dataset=self.peptides_dataset
										)

		assert train_loss["error"] < train_err_tol
		assert val_loss["error"] < test_err_tol
		assert test_loss["error"] < test_err_tol

	def test_corrected_mse(self):
		""" 
		Tests the model's ability to train, then reconstruct the input 
		matrix within a reasonable error tolerance. 

		For variance-corrected MSE loss function
		"""
		train_loss, val_loss, test_loss = train_model(
											loss="CMSE", 
											dataset=self.peptides_dataset
										)


		assert train_loss["error"] < train_err_tol
		assert val_loss["error"] < test_err_tol
		assert test_loss["error"] < test_err_tol

