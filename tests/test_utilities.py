"""
TEST UTILITIES

Testing the functions in the util_functions_test module.
Not sure if this module is needed, necessarily
"""
import numpy as np
import pandas as pd
import torch
import unittest
import pytest

import util_functions_test

class UtilitiesTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """

	@pytest.mark.skip(reason="not implemented")
	def test_mse(self):
		"""
		Make sure the mse loss function is working
		"""
		assert True == True

	@pytest.mark.skip(reason="not implemented")
	def test_rmse(self):
		"""
		Make sure the relative mse loss function is working
		"""
		assert True == True

	@pytest.mark.skip(reason="not implemented")
	def test_simulated_mat(self):
		"""
		Make sure the simulated matrices look ok
		"""
		assert True == True

	@pytest.mark.skip(reason="not implemented")
	def test_simulated_realistic(self):
		"""
		Make sure the more realistic simulated 
		matrices look ok
		"""
		assert True == True

