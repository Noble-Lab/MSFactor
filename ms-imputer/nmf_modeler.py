"""
NMF-MODELER

This module defines two generic classes, NMFModeler and Dataset.
NMFModeler class allows you to generate modeler objects, train, get 
intermediate plots, get reconstruction errors. Dataset class 
initalizes a dataset object, allows you to partition data and
log transform. 
"""
import numpy as np
import pandas as pd
import yaml
import torch

from models.linear import GradNMFImputer

import generate_plots
import util_functions

class NMFModeler:
	""" 
	Generic modeler class for NMF models

	Parameters
	----------
	log_transform : boolean, log transform the input matrix? 
	tolerance : float, the convergence tolerance, for evaluating early 
				stopping.
	max_epochs : int, max number of training iterations. 
	batch_size : int, size of each training and evaluation batch. 
	learning_rate : float, the SGD step size. 
	"""
	def __init__(
		self, 
		tolerance=0.0001,
		max_epochs=3000,
		batch_size=1000,
		learning_rate=0.05,
	):
		self.tolerance = tolerance
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		
		self.fitted = False
		self.model = None
		self.recon = None

		# assign the evaluation loss function
		self.loss_func_eval = util_functions.mse_func_np

	def fit(
		self,
		pxd,
		train, 
		val, 
		n_factors=8 # default number of latent factors 
	):
		""" 
		Fit an NMF model, for a single input matrix
		
		Parameters
		----------
		train : np.ndarray, the input (training) matrix, to be transformed
		val : np.ndarray, the validation matrix, optional
		n_factors : int, the number of latent factors to use for reconstruction,
						optional
		pxd : str, the PRIDE dataset identifier

		Returns
		-------
		none
		"""
		nmf_model = GradNMFImputer(
							train.shape[0], 
							train.shape[1], 
							n_factors=n_factors, 
							stopping_tol=self.tolerance, 
							train_batch_size=self.batch_size, 
							eval_batch_size=self.batch_size,
							n_epochs=self.max_epochs, 
							loss_func="MSE",
							optimizer=torch.optim.Adam,
							optimizer_kwargs={"lr": self.learning_rate}
					)
		
		# fit and transform
		recon = nmf_model.fit_transform(train, val)

		self.fitted = True
		self.recon = recon
		self.model = nmf_model
		return

class Dataset:
	""" 
	The dataset class
	
	Parameters
	----------
	pxd : str, the PRIDE identifier
	data_pth : PosixPath, path to the pre-processed peptide/
				protein quants matrix
	"""
	def __init__(self, pxd, data_pth):
		self.pxd = pxd
		self.quants_matrix = pd.read_csv(data_pth)
		self.quants_matrix.replace(0, np.nan, inplace=True)
		self.quants_matrix = np.array(self.quants_matrix)

		self.partitioned = False

		self.train = None
		self.val = None
		self.test = None

	def partition(self, val_frac=0.1, test_frac=0.1, min_present=10):
		""" 
		Partition the protein/peptide quants matrix into disjoint
		training, validation and test sets
		
		Parameters
		----------
		val_frac : float, the portion of the dataset to designate as 
					validation set
		test_frac : float, the portion of the dataset to designate as test
		min_present : int, the minimum number of present values for 
					each row in the training set

		Returns
		-------
		none
		"""

		train, val, test = util_functions.split(self.quants_matrix, 
									val_frac=val_frac,
									test_frac=test_frac, 
									min_present=min_present
								)

		self.partitioned = True
		
		self.train = train
		self.val = val
		self.test = test

		return

