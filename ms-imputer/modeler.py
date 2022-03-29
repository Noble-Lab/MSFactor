"""
MODELER

This module defines two generic classes, Modeler and Dataset.
Modeler class allows you to generate modeler objects, train, get 
intermediate plots, get reconstruction errors. Dataset class 
initalizes a dataset object, allows you to partition data and
log transform. 
"""
import numpy as np
import pandas as pd
import yaml

import modeler_utils
import intermediate_plots
import util_functions

class Modeler:
	""" 
	Generic modeler class for NMF, kNN and MissForest models
	
	Parameters
	----------
	model_type : str, what type of model? 
				NMF, MissForest, kNN
	loss_func : str, what loss function to use. 
				NMF only. 
	log_transform : boolean, log transform the input matrix? 
				NMF only. 
	tolerance : float, the convergence tolerance, for evaluating early 
				stopping. NMF only 
	max_epochs : int, max number of training iterations. 
				NMF & MissForest
	batch_size : int, size of each training and evaluation batch. 
				NMF only
	learning_rate : float, the SGD step size. 
				NMF only
	parallelize : str, ["forests", "variables"], parallelization method. 
				MissForest only. 
	"""
	def __init__(
		self, 
		model_type, 
		loss_func=None,
		log_transform=False,
		tolerance=0.0001,
		max_epochs=2000,
		batch_size=1000,
		learning_rate=0.05,
		parallelize="forests",
	):
		self.type = model_type
		self.loss_func = loss_func
		self.log_transform = log_transform
		self.tolerance = tolerance
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.parallelize = parallelize
		
		self.fitted = False
		self.model = None
		self.recon = None

		# assign the evaluation loss function
		if self.loss_func == "RMSE":
			self.loss_func_eval = util_functions.rmse_func_np
		else:
			self.loss_func_eval = util_functions.mse_func_np

	def fit(self, train_mat, valid_mat, n_factors, pxd):
		""" 
		Fit the model 
		
		Parameters
		----------
		train_mat : np.ndarray, the dataset to be fit (trining set)
		valid_mat : np.ndarray, the validation set
		n_factors : int, the number of factors to use for training
		pxd : str, the PRIDE identifier

		Returns
		------
		recon : np.ndarray, the reconstructed matrix
		"""
		if self.type == "NMF":
			recon, model = modeler_utils.fit_nmf(
											pxd,
											train_mat, 
											valid_mat, 
											n_factors,
											self.tolerance, 
											self.batch_size,
											self.max_epochs,
											self.loss_func,
											self.learning_rate
										)
			self.model = model
		elif self.type == "KNN":
			recon = modeler_utils.fit_knn(train_mat, n_factors)
		elif self.type == "MissForest":
			recon = modeler_utils.fit_missForest(
											pxd,
											train_mat, 
											n_factors, 
											self.max_epochs,
											self.parallelize
										)
		else:
			raise ValueError("Unrecognized model type")

		self.fitted = True
		self.recon = recon

		return recon

	def get_loss_curves(self, n_factors, pxd, tail=None):
		""" 
		Plot training and validation loss curves. Really only set up for
		NMF models. Saves plot as a .png to ./results/ directory

		Parameters
		----------
		n_factors : int, the number of factors used during training
		pxd : str, the PRIDE identifier
		tail: str, the config file to associate with this loss curve.
				OPTIONAL

		Returns
		-------
		none
		"""
		if self.fitted and self.type == "NMF":
			intermediate_plots.plot_train_loss(
										self.model, 
										pxd, 
										n_factors, 
										n_col_factors=None, 
										model_type=self.type, 
										eval_loss=self.loss_func,
										tail=tail
								)
		if not self.fitted:
			raise ValueError("Model must be fit first")
		if self.type != "NMF":
			raise ValueError("The function only set up for NMF models")
		
		return

	def get_correlation_plots(self, valid_mat, n_factors, pxd, tail=None):
		"""
		Plots real vs imputed abundances and calculates Pearson 
		correlation coefficient. Saves plot as .png to ./results/

		Parameters
		----------
		valid_mat : np.ndarray, the validation matrix
		n_factors : int, the number of factors used during training
		pxd : str, the PRIDE identifier
		tail: str, the config file to associate with this loss curve.
				OPTIONAL

		Returns
		-------
		none
		"""
		if self.fitted and self.type != "KNN":
			intermediate_plots.real_v_imputed_basic(
									self.recon, 
									valid_mat, 
									pxd, 
									n_factors, 
									col_factors=None, 
									model_type=self.type,
									log_transform=self.log_transform,
									tail=tail
								)

		if not self.fitted:
			raise ValueError("Model must be fit first")
		
		return

	def get_sanity_check_plots(self, n_factors, pxd, tail=None):
		""" 
		Get sanity check plots that zoom in on the last 50 iters
		of model training and make sure convergence criteria is 
		doing what its supposed to be doing. 

		Parameters
		----------
		n_factors : int, the number of factors used during training
		pxd : str, the PRIDE identifier
		tail: tail, the config file to associate with this loss curve.
			OPTIONAL
		
		Returns
		-------
		none
		"""
		if self.fitted and self.type == "NMF":
			intermediate_plots.sanity_check_loss_curve(
									self.model, 
									pxd, 
									n_factors, 
									col_factors=None, 
									model_type=self.type,
									tail=tail
								)
			intermediate_plots.sanity_check_boxplot(
									nmf_model, 
									pxd, 
									n_factors, 
									col_factors=None, 
									model_type="NMF",
									tail=tail
								)

		if not self.fitted:
			raise ValueError("Model must be fit first")
		return

	def collect_errors(
			self, 
			train_mat, 
			valid_mat, 
			test_mat, 
			recon_mat, 
			n_factors
	):
		"""
		Collect the error, for training, validation and test sets
		
		Parameters
		----------
		train_mat, valid_mat, test_mat : np.ndarray, the training,
										 validation and test set matrices 
		recon_mat : np.ndarray, the reconstructed matrix
		n_factors : int, the number of factors used during training

		Returns
		-------
		train_res, val_res, test_res : np.ndarray, the train, validation 
										and test set error. 
		"""
		if not self.fitted:
			raise ValueError("Model has not been fit")

		train_mse = self.loss_func_eval(train_mat, recon_mat)
		valid_mse = self.loss_func_eval(valid_mat, recon_mat)
		test_mse = self.loss_func_eval(test_mat, recon_mat)

		train_res = {
			"error": train_mse,
			"n_factors": n_factors,
			"split": "Train",
			}
		val_res = {
			"error": valid_mse,
			"n_factors": n_factors,
			"split": "Validation",
			}
		test_res = {
			"error": test_mse,
			"n_factors": n_factors,
			"split": "Test",
			}

		return train_res, val_res, test_res

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
		self.partitioned = False
		self.logged = False

		self.train = None
		self.val = None
		self.test = None

	def log_transform(self):
		""" 
		Log transform the protein/peptide quants matrix.
		In place transformation. 
		"""
		#self.quants_matrix = np.log(self.quants_matrix)
		self.quants_matrix = np.log10(self.quants_matrix)
		self.logged = True

		return

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

