"""
TEST-BASIC
"""
import unittest
import pytest
import pandas as pd 
import numpy as np
import torch

from ms_imputer.models.linear import GradNMFImputer
import ms_imputer.util_functions

# simulated matrix configs
rng = np.random.default_rng(42) # random seed
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)

# training params
n_factors = 4 
tolerance = 0.0001
batch_size = 100
max_iters = 400
learning_rate = 0.05
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

class FastTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init simulated matrix, write to csv
		matrix = simulate_matrix(matrix_shape)
		pd.DataFrame(matrix).to_csv("simulated.csv", index=False)

		self.train, self.val, self.test = split(
											matrix,
											val_frac=0.1, 
											test_frac=0.1, 
											min_present=1
										)

	def test_nmf_imputer(self):
		""" 
		Tests GradNMFImputer class's ability to train an 
		NMF model, and reconstruct the input matrix 
		within a reasonable error tolerance. 
		"""
		nmf_model = GradNMFImputer(
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

		recon = nmf_model.fit_transform(self.train, self.val)

		train_err = mse_func_np(self.train, recon)
		val_err = mse_func_np(self.val, recon)
		test_err = mse_func_np(self.test, recon)

		assert train_err < train_err_tol
		assert val_err < test_err_tol
		assert test_err < test_err_tol

		assert True == True






