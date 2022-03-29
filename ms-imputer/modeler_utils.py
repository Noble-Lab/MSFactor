"""
MODELER_UTILS

Utility functions for the Modeler class. Allows you
to fit NMF models
"""
from models.linear import GradNMFImputer

import torch
import numpy as np

def fit_nmf(
	PXD, 
	train_mat, 
	val_mat, 
	n_factors, 
	tolerance, 
	batch_size,
	max_epochs, 
	loss_func, 
	learning_rate,
):
	""" 
	Fit an NMF model, for a single PRIDE matrix
	
	Parameters
	----------
	PXD : str, the PRIDE dataset identifier
	train_mat : np.ndarray, the input (training) matrix, to be transformed
	val_mat : np.ndarray, the validation matrix, included for loss 
				function plotting
	n_factors : int, the number of latent factors to use for reconstruction
	tolerance : float, the convergence tolerance, for evaluating early stopping
	batch_size : int, max number of training iterations. 
	max_epochs : int, size of each training and evaluation batch. 
	loss_func : str, the loss function to use {"MSE", "RMSE"}
	learning_rate : float, the SGD step size.

	Returns
	-------
	recon: np.ndarray, the reconstructed matrix
	"""
	nmf_model = GradNMFImputer(train_mat.shape[0], 
								train_mat.shape[1], 
								n_factors=n_factors, 
								stopping_tol=tolerance, 
								train_batch_size=batch_size, 
								eval_batch_size=batch_size,
								n_epochs=max_epochs, 
								loss_func=loss_func,
								optimizer=torch.optim.Adam,
								optimizer_kwargs={"lr": learning_rate})
	
	# fit and transform
	recon = nmf_model.fit_transform(train_mat, val_mat)
	return recon, nmf_model

