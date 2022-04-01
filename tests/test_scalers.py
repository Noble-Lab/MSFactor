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
import util_functions_test

# simulated matrix configs
matrix_shape = (12,10,3) # (n_rows, n_cols, rank)
min_present = 1

# error assessment params
err_tol = 0.1

class ScalerTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

		# init a basic simulated matrix
		self.matrix = util_functions_test.simulate_matrix_realistic(matrix_shape)

		train, val, test = util_functions_test.split(
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

