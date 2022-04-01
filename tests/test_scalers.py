"""
TEST-SCALERS

This module tests the scalers.
"""
import unittest
import pytest
import os
import pandas as pd 
import numpy as np
import torch

from ms_imputer.models.scalers import StandardScaler
import ms_imputer.util_functions

# simulated matrix configs
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)
min_present = 1

# error assessment params
err_tol = 0.1

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

class ScalerTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init the first (basic) simulated matrix
		self.matrix = simulate_matrix_realistic(matrix_shape)

		train, val, test = ms_imputer.util_functions.split(
									self.matrix,
									val_frac=0.1, 
									test_frac=0.1, 
									min_present=min_present
		)

		self.train = torch.tensor(train)
		self.val = torch.tensor(val)
		self.test = torch.tensor(test)

	def test_standard_scaler(self):
		""" 
		Test the standard scaler. Make sure values are actually
		being divided by the std of the matrix.   
		"""
		# with StandardScaler
		scaler = StandardScaler()
		std_scaled = scaler.fit_transform(self.train)

		# manually scale
		scale_factor = np.nanstd(self.train)
		manual_scaled = self.train / scale_factor
		manual_scaled = torch.tensor(manual_scaled)

		std_scaled_nonmissing = std_scaled[~torch.isnan(std_scaled)]
		manual_scaled_nonmissing = manual_scaled[~torch.isnan(manual_scaled)]

		assert np.all(np.isclose(std_scaled_nonmissing, manual_scaled_nonmissing, atol=0.1))

