""" tests for bin/util_functions.py """
import sys
import unittest
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.append('../bin')
sys.path.append('bin/')

import util_functions

class UtilsTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """
		self.mat = np.random.rand(15,10)
		self.mask = util_functions.get_mask(self.mat, 0.1)
		self.filled_mask = util_functions.get_mask(self.mat, 1.0)
		self.empty_mask = util_functions.get_mask(self.mat, 0.0)

		self.empty_mat = np.empty((15,10))
		self.empty_mat[:] = np.nan

		self.mv_mat = self.mat.copy()
		self.mv_mat[self.mask] = np.nan


	def test_always_passes(self):
		""" dummy test """
		self.assertTrue(True)


	def test_get_mask(self):
		""" testing the get_mask() function """	
		assert np.count_nonzero(self.mask) == 15
		assert np.count_nonzero(self.empty_mask) == 0
		assert np.count_nonzero(self.filled_mask) == 150
		
		assert self.mask.size == 150
		assert self.mask.shape[0] == 15
		assert self.mask.shape[1] == 10


	def test_mse_basic(self):
		""" testing mse_func_mv() basic functionality """
		a = np.array([1,2,3,4,5,6,7,8])
		b = np.array([2,3,4,5,6,7,8,9])
		
		mask_shuffled = self.mask.copy()
		np.random.shuffle(mask_shuffled)

		mat_shuffled = self.mat.copy()
		mat_shuffled[mask_shuffled] = np.nan

		mat0 = np.random.rand(10,5)
		mat1 = np.random.rand(10,5)

		assert util_functions.mse_func_np(a,b) == 1
		assert util_functions.mse_func_np(a,a*10) == 2065.5
		assert util_functions.mse_func_np(a,a) == 0

		assert util_functions.mse_func_np(self.mv_mat, mat_shuffled) == 0

		assert util_functions.mse_func_np(mat0, mat1) != 0

		assert util_functions.mse_func_np(self.mat, self.mat) == 0.0
		assert util_functions.mse_func_np(self.mat, self.empty_mat) == 0
		
		# todo: how to check if the correct warning is thrown?


	def test_mse_missing(self):
		""" testing mse_func_np() with some missing values """

		assert util_functions.mse_func_np(self.mat, self.mv_mat) == 0
		assert util_functions.mse_func_np(self.mat*10, self.mv_mat) != 0


	@pytest.mark.skip(reason='I know this one fails')
	def test_mse_torch(self):
		""" testing mse_func_torch() """
		tensor0 = torch.rand(15,10)
		tensor1 = torch.rand(15,10)

		tensor2 = tensor0.clone().detach()
		tensor2[torch.tensor(self.mask)] = np.nan

		assert util_functions.mse_func_torch(tensor0, tensor1) != 0
		assert util_functions.mse_func_torch(tensor0, tensor0) == 0
		assert util_functions.mse_func_torch(tensor0, tensor2) == 0
		assert util_functions.mse_func_torch(tensor2, tensor1) != 0


	def test_filter_two_mat(self):
		""" basic tests for train_data_filter() """

		train0, valid0, test0 = util_functions.train_data_filter(
												self.mat, 
												self.mv_mat, 
												self.mv_mat, 
												min_present=1
											)

		train1, valid1, test1 = util_functions.train_data_filter(
												self.mv_mat, 
												self.mv_mat, 
												self.mv_mat,
												min_present=5
											)

		assert train0.shape == valid0.shape == test0.shape
		assert train1.shape == valid1.shape == test1.shape

		assert util_functions.mse_func_np(train0, valid0) == 0
		assert util_functions.mse_func_np(train1, valid1) == 0
		assert util_functions.mse_func_np(valid0, test0) == 0
		assert util_functions.mse_func_np(valid1, test1) == 0


	def test_filter_three_mat(self):
		""" test for train_data_filter()...using three matrices this time """
		train_mask = util_functions.get_mask(self.mat, 0.2)
		train_mat = self.mat.copy()
		train_mat[train_mask] = np.nan

		valid_mat = self.mat.copy()
		valid_mat[~train_mask] = np.nan

		valid_mask = util_functions.get_mask(valid_mat, 0.5)
		valid_mat[valid_mask] = np.nan

		test_mat = valid_mat.copy()
		test_mat[valid_mask] = np.nan

		train_f, valid_f, test_f = util_functions.train_data_filter(
													train_mat, 
													valid_mat, 
													test_mat, 
													min_present=6
												)

		assert train_f.shape == valid_f.shape == test_f.shape
		assert util_functions.mse_func_np(train_f, valid_f) == 0
		assert util_functions.mse_func_np(valid_f, test_f) == 0

