""" 
UTIL_FUNCTIONS

Various utility functions for wrangling peptide quant 
files into the appropriate format, and for evaluating and 
testing our imputation models thereafter. 

Can import like so: 

import sys
sys.path.append('path/to/bin')
from util_functions import *
"""
import numpy as np
import pandas as pd
import torch
from itertools import product
import json
import logging

#from scipy.optimize import curve_fit

LOGGER = logging.getLogger(__name__)
RAW_PATH = 'data/raw/'
TRIM_PATH = 'data/trim/'

def trim_raw_data(pxd_paths, PXDs):
    """ 
    Trims a raw maxquant outfile to include only the 
    protein quant columns. Writes trimmed dataframes 
    to csv. 
    
    Parameters
    ----------
    pxd_paths : list of PosixPaths, to untrimmed csvs 
    PXDs : list of strs, PRIDE identifier corresponding to the
                specified paths
    Returns
    -------
    None
    """
    for pxd_path, PXD in zip(pxd_paths, PXDs):
        # "dtype=object" indicates mixed data type
        df_raw = pd.read_csv(pxd_path, sep='\t', dtype=object)

        # identify just the protein quant columns
        intensity_cols = []
        for x in list(df_raw.columns):
            if 'Intensity' in x:
                intensity_cols.append(x)

        # subset
        df_trim = df_raw[intensity_cols]

        # drop these summary cols
        if 'Intensity' in df_trim.columns:
            df_trim = df_trim.drop(['Intensity'], axis=1)
        if 'Intensity L' in df_trim.columns:
            df_trim = df_trim.drop(['Intensity L'], axis=1)
        if 'Intensity H' in df_trim.columns:
            df_trim = df_trim.drop(['Intensity H'], axis=1)

        # write to csv
        df_trim.to_csv(TRIM_PATH + PXD + "_peptides.csv", index=None)

    return 


def read_projects(json_path):
    """ 
    Reads in the input json file

    Parameters
    ----------
    json_path : PosixPath, to the project names json file

    Returns
    -------
    Dictionary of {projects : filename} pairs
    """
    f = open(json_path)
    projects = json.load(f)

    return projects["filenames"]


def get_mask(mat, missing_frac):
    """ 
    Masking function for a specified proportion of 
    values, for an input matrix
    
    Parameters
    ----------
    mat : array-like, the matrix to be masked
    missing_frac : float, the proportion of values to be masked

    Returns
    -------
    mask_ : boolean array, the mask
    """
    raveled = mat.ravel()
    num_holdout = int(missing_frac * len(raveled))

    mask_ = np.full(len(raveled), False)
    mask_[:num_holdout] = True
    np.random.shuffle(mask_)

    mask_ = np.reshape(mask_, mat.shape)

    return mask_


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


def mse_func_torch(X, Y):
    """ 
    Calculate the MSE for two pytorch tensors with missing values. 
    Same as above, but for PyTorch tensors.
    
    Parameters
    ----------
    X : torch.tensor, the first tensor
    Y : torch.tensor, the second tensor
    
    Returns
    -------
    float, the mean squared error between values present across 
                both tensors
    """
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    missing = torch.isnan(X) | torch.isnan(Y)
    mse = torch.sum(X[~missing] - Y[~missing]**2)
    
    return mse / torch.sum(~missing)


def rmse_func_torch(x, y):
    """ 
    Gets the Relative Mean Squared Error between 
    two pytorch tensors
    
    Parameters
    ----------
    x : array-like, the predicted (imputed) values
    y : array-like, the target (ground truth) values
        
    Returns
    -------
    rmse : float, the Relative Mean Squared Error
    """
    x_flat = torch.ravel(x)
    y_flat = torch.ravel(y)

    missing = torch.isnan(x_flat) | torch.isnan(y_flat)

    rmse = torch.sum((y_flat[~missing] - x_flat[~missing])**2 / 
                            y_flat[~missing]**2)

    return rmse


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


def train_data_filter(train_mat, valid_mat, test_mat=None, min_present=10):
    """ 
    If a row is contains above a certain number / percentage of MVs, 
    we need to remove it from train, valid and test datasets.
    Ensures that all three matrices will have the same rows.  
    
    Parameters
    ----------
    train_mat, valid_mat, test_mat : np.ndarray, unfiltered matrices 
        corresponding to the train, valid and test sets. 
        test_mat is OPTIONAL
    min_present : float, the minimum number of present valeus to 
            require for each row

    Returns
    -------
    train_mat_filt, valid_mat_filt, test_mat_filt : np.ndarray, 
            filtered matrices for the train, valid and test sets
    """
    # to pandas
    train_mat_pd = pd.DataFrame(train_mat)
    valid_mat_pd = pd.DataFrame(valid_mat)
    test_mat_pd = pd.DataFrame(test_mat)

    # How many zeros are we allowed?
    max_zeros = train_mat_pd.shape[1] - min_present
    print(f"Eliminating train rows containing >={max_zeros} missing values.")
    
    # train_data is the only one we need to make sure has no all MV rows
    zeros_by_row = np.isnan(train_mat_pd).sum(axis=1)
    print(f"Reducing from {train_mat_pd.shape[0]} to {(zeros_by_row < max_zeros).sum()} rows.")

    # need both train and test matrices to contain the same rows
    train_mat_pd_filt = train_mat_pd[zeros_by_row < max_zeros]
    valid_mat_pd_filt = valid_mat_pd[zeros_by_row < max_zeros]
    test_mat_pd_filt = test_mat_pd[zeros_by_row < max_zeros]
    
    # convert back to np
    train_mat_filt = train_mat_pd_filt.to_numpy()
    valid_mat_filt = valid_mat_pd_filt.to_numpy()
    test_mat_filt = test_mat_pd_filt.to_numpy()
    
    return train_mat_filt, valid_mat_filt, test_mat_filt


def matrix_downsample(mat, min_present=10):
    """ 
    For a single matrix, remove the rows that have too many missing values

    TODO: modify this to tolerate both np arrays and pandas dataframes

    Parameters
    ----------
    mat : np.ndarray, or pd.DataFrame, the input matrix 
    min_present : int, the minimum number of present values 
                    that a row must have in order to keep it
    Returns
    -------
    np.ndarray, the filtered matrix
    """
    zeros_by_row = ((mat == 0).sum(axis=1))
    return mat[zeros_by_row < min_present]


def get_combinations(hlist, truncate=True):
    """ 
    Get all pairwise combinations in a list of
    hyperparameters. Given the truncate param, 
    return only the (x,y) pairs such that y >= x.
    (x,y) indicates (n row lf, n col lf)
    
    Parameters
    ----------
    hlist : list, of hyperparameters 
    truncate : boolean
        false--do full 2D grid hyperparameter search, or 
        true--truncate, including only the (x,y) combinations 
                such that x <= y
    
    Returns
    -------
    filt_combs : list of tuples, pairwise combinations
    """
    combs = list(product(hlist, repeat=2))
    combs = np.array(combs)
    filt_combs = combs[[pair[0] <= pair[1] for pair in combs]]

    if truncate: # truncated 2D grid search
        return filt_combs
    if not truncate: # full 2D grid search
        return combs


def prob_select(elm):
    """ 
    For a given float (0->1), take a random float (0->1).
    If the input float is larger, return True, else return False
    
    Parameters
    ----------
    elm : float, a matrix element  

    Returns
    -------
    boolean
    """
    seed = np.random.uniform(0,1)
    
    if elm > seed:
        return True
    else:
        return False


# def sigmoid_func(x, x0, k):
#     """ 
#     Defines sigmoid helper function to use for curve_fit() mapping
#     """
#     return 1 / (1 + np.exp(-k*(x-x0)))


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

    if num_discard > 0:
        LOGGER.info(
            "Discarding %i of %i proteins with too many missing values.",
            num_discard,
            matrix.shape[0],
        )

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

    if num_discard > 0:
        LOGGER.info(
            "Discarding %i of %i proteins with too many missing"
            " values in the training set.",
            num_discard,
            train_set.shape[0]
        )

    train_set = np.delete(train_set, discard, axis=0)
    val_set = np.delete(val_set, discard, axis=0)
    test_set = np.delete(test_set, discard, axis=0)

    return train_set, val_set, test_set
