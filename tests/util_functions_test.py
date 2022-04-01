""" 
UTIL_FUNCTIONS_TEST

Utility functions to help with testing and evaluation 
of the model code. 
"""
import numpy as np
import pandas as pd
import torch
rng = np.random.default_rng(42) # random seed

from ms_imputer.models.linear import GradNMFImputer

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

def rmse_func_np(x, y):
    """ 
    Gets the Relative Mean Squared Error between 
    two numpy arrays

    Parameters
    ----------
    x : np.ndarray, the predicted (imputed) values
    y : np.ndarray, the target (ground truth) values
    
    Returns
    -------
    rmse : float, the Relative Mean Squared Error
    """
    x_flat = x.ravel()
    y_flat = y.ravel()

    missing = np.isnan(x_flat) | np.isnan(y_flat)

    if (np.sum(~missing) == 0):
        print("Warning: Computing relative MSE from all missing values.")
        return 0

    num = np.sum((y_flat[~missing] - x_flat[~missing])**2)
    denom = np.sum(y_flat[~missing]**2)
    rmse = num / denom

    return rmse

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

def split(matrix, val_frac=0.1, test_frac=0.1, min_present=5, 
            random_state=42):
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
    test_frac : float, optional
        The fraction of data to assign to the test set.
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
    test_set : numpy.ndarray
        The test set, where other values are NaNs.
    """
    rng = np.random.default_rng(random_state)
    if val_frac + test_frac > 1:
        raise ValueError("'val_frac' and 'test_frac' cannot sum to more than 1.")

    # Prepare the matrix:
    matrix = np.array(matrix).astype(float)
    matrix[matrix == 0] = np.nan
    num_present = np.sum(~np.isnan(matrix), axis=1)
    discard = num_present < min_present
    num_discard = discard.sum()

    matrix = np.delete(matrix, discard, axis=0)

    # Assign splits:
    indices = np.vstack(np.nonzero(~np.isnan(matrix)))
    rng.shuffle(indices, axis=1)

    n_val = int(indices.shape[1] * val_frac)
    n_test = int(indices.shape[1] * test_frac)
    n_train = indices.shape[1] - n_val - n_test

    train_idx = tuple(indices[:, :n_train])
    val_idx = tuple(indices[:, n_train:(n_train + n_val)])
    test_idx = tuple(indices[:, -n_test:])

    train_set = np.full(matrix.shape, np.nan)
    val_set = np.full(matrix.shape, np.nan)
    test_set = np.full(matrix.shape, np.nan)

    train_set[train_idx] = matrix[train_idx]
    val_set[val_idx] = matrix[val_idx]
    test_set[test_idx] = matrix[test_idx]

    # Remove Proteins with too many missing values:
    num_present = np.sum(~np.isnan(train_set), axis=1)
    discard = num_present < min_present
    num_discard = discard.sum()

    train_set = np.delete(train_set, discard, axis=0)
    val_set = np.delete(val_set, discard, axis=0)
    test_set = np.delete(test_set, discard, axis=0)

    return train_set, val_set, test_set

