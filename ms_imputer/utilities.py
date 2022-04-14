"""
UTILITIES

ms_imputer utilities
"""
import pandas as pd
import numpy as np
import random
rng = np.random.default_rng(42) # random seed

def maxquant_trim(maxquant_path, out_stem):
	"""
	Keep only the peptide/protein "Intensity" columns from 
	Maxquant's proteinGroups.txt and peptides.txt standard
	output file formats. Writes new file to specified path

	Parameters
	----------
	maxquant_path : str, posix.path to maxquant proteinGroups.txt
					or peptides.txt output file
	out_stem : str, output file stem
	
	Returns
	-------
	0, 1 : bool, whether or not the input file was actually trimmed
	"""
	raw_maxquant = pd.read_csv(maxquant_path, sep="\t", dtype=object)

	# identify just the protein/peptide quant columns
	intensity_cols = []
	for x in list(raw_maxquant.columns):
		if 'Intensity' in x:
			intensity_cols.append(x)

	# already trimmed
	if len(raw_maxquant.columns) == len(intensity_cols):
		return 0

	# subset
	df_trim = raw_maxquant[intensity_cols]

	# drop these summary cols
	if 'Intensity' in df_trim.columns:
		df_trim = df_trim.drop(['Intensity'], axis=1)
	if 'Intensity L' in df_trim.columns:
		df_trim = df_trim.drop(['Intensity L'], axis=1)
	if 'Intensity H' in df_trim.columns:
		df_trim = df_trim.drop(['Intensity H'], axis=1)

	# write to csv
	df_trim.to_csv(out_stem + "_quants.csv", index=None)

	# trimmed
	return 1

def mse_func_np(x_mat, y_mat):
	"""
	Calculate the MSE for two matricies with missing values. Each
	matrix can contain MVs, in the form of np.nans

	Parameters
	----------
	x_mat : np.ndarray, the first matrix 
	y_mat : np.ndarray, the second matrix

	Returns
	-------
	float, the mean squared error between values present 
	        across both matrices
	"""
	x_rav = x_mat.ravel()
	y_rav = y_mat.ravel()
	missing = np.isnan(x_rav) | np.isnan(y_rav)
	mse = np.sum((x_rav[~missing] - y_rav[~missing])**2)

	if (np.sum(~missing) == 0):
		print("Warning: Computing MSE from all missing values.")
		return 0
	return mse / np.sum(~missing)

def kfolds_partition(matrix, split_indices, k, min_present=4):
	""" 
	Given a list of present value indices and a matrix, 
	generates train, validation and test sets. 
	
	Parameters
	----------
	matrix : np.ndarray, the input matrix
	split_indices : np.ndarray, a nested list, outputted from 
	                shuffle_and_split(), of the indices of present
	                values
	k : int, the current fold. This will become the index of the
					validation set
	min_present : int, the minumum number of present values per 
					row in original matrix
	Returns
	-------
	train_set, valid_set: np.ndarray, the training and 
					validation sets
	"""
	# init validation set
	valid_set = np.full(matrix.shape, np.nan)

	valid_indices = split_indices[k]

	# unpackage zipped array, get tuples
	valid_rows, valid_cols = zip(*valid_indices)
    
	# add values to validation set
	valid_set[valid_rows, valid_cols] = matrix[valid_rows, valid_cols]

	# get validation and test masks
	valid_mask = np.isnan(valid_set)

	# init train set, set validation and test masks to nan
	train_set = matrix.copy()
	train_set[~valid_mask] = np.nan

	# remove peptides/proteins with too many missing values:
	num_present = np.sum(~np.isnan(train_set), axis=1)
	discard = num_present < min_present
	num_discard = discard.sum()

	train_set = np.delete(train_set, discard, axis=0)
	valid_set = np.delete(valid_set, discard, axis=0)

	return train_set, valid_set

def discard_empty_rows(matrix, min_present=4):
	"""
	Trim the input matrix such that rows with < min_present
	present values are discarded. 

	Parameters
	----------
	matrix : np.ndarray, the input matrix
	min_present : int, if a row has fewer present values than
					this, discard it
	Returns
	-------
	matrix : np.ndarray, the trimmed matrix
	discard : array-like, the indices of the discarded rows
	"""
	matrix = np.array(matrix).astype(float)
	matrix[matrix == 0] = np.nan
	num_present = np.sum(~np.isnan(matrix), axis=1)
	discard = num_present < min_present
	num_discard = discard.sum()

	#matrix = np.delete(matrix, discard, axis=0)

	matrix_pd = pd.DataFrame(matrix)
	matrix_pd = matrix_pd[~discard]

	return matrix_pd, discard

def k_folds_split(matrix, k_folds):
	""" 
	Shuffle the indices of present values from an input 
	matrix, and then split into k_folds equal pieces. Return
	indices of these pieces/

	Parameters
	----------
	matrix : np.ndarray, the input matrix
	k_folds : int, the number of folds to split into

	Returns
	-------
	split_indices : np.ndarray, a nested list containing the
				randomly selected indicies of present values
				only, for each of k splits
	"""
	# get indices from matrix, shuffle
	indices = np.vstack(np.nonzero(~np.isnan(matrix)))
	rng.shuffle(indices, axis=1)

	# zip, into array 
	indices_shuffled_zip = np.array(list(zip(indices[0],indices[1])))

	# split shuffled indices into k_folds equal pieces
	split_indices = np.array_split(indices_shuffled_zip, k_folds)

	return split_indices
 
def partition(matrix, val_frac=0.1, min_present=4, random_state=42):
	"""
	Split a data matrix into training, validation, test sets.

	Note that the fractions of data in the validation and tests 
	sets is only approximate due to the need to drop rows with 
	too much missing data.

	Parameters
	----------
	matrix : array-like
		The data matrix to split.
	val_frac : float, optional
		The fraction of data to assign to the validation set.
	min_present : int, optional
		The minimum number of non-missing values required in each 
		row of the training set.
	random_state : int or numpy.random.Generator
		The random state for reproducibility.

	Returns
	-------
	train_set : numpy.ndarray
		The training set.
	val_set : numpy.ndarray
		The validation set, where other values are NaNs.
	"""
	rng = np.random.default_rng(random_state)

	# assign splits:
	indices = np.vstack(np.nonzero(~np.isnan(matrix)))
	rng.shuffle(indices, axis=1)

	n_val = int(indices.shape[1] * val_frac)
	n_train = indices.shape[1] - n_val

	train_idx = tuple(indices[:, :n_train])
	val_idx = tuple(indices[:,-n_val:])

	train_set = np.full(matrix.shape, np.nan)
	val_set = np.full(matrix.shape, np.nan)

	train_set[train_idx] = matrix[train_idx]
	val_set[val_idx] = matrix[val_idx]

	# remove peptides/proteins with too many missing values:
	num_present = np.sum(~np.isnan(train_set), axis=1)
	discard = num_present < min_present
	num_discard = discard.sum()

	train_set = np.delete(train_set, discard, axis=0)
	val_set = np.delete(val_set, discard, axis=0)

	return train_set, val_set

