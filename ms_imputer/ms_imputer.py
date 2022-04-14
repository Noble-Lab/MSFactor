""" 
MS-IMPUTER

The commandline entry point for ms_imputer.  
"""
import pandas as pd 
import numpy as np
import click
import torch
import os
from tqdm import tqdm
import warnings

from ms_imputer.models.linear import GradNMFImputer
import ms_imputer.utilities

# not a good long term solution
warnings.filterwarnings(
	action="ignore",
	category=UserWarning,
	module="torch")

def cross_validate(matrix, folds, grid, tol, b_size, n_epochs, lr):
	""" 
	Run nested cross validation for an input matrix, a given
	hyperparameter grid and a given value of k. For each of k folds, 
	partition into train and validate, train NMF models across 
	every hparam. Record train and validation errors. Return a 
	dataframe of per-hyperparameter averages, across each of k folds
	
	Parameters
	----------
	matrix : array-like, 2D, the input matrix 
	folds : int, the number of folds
	grid : list, the 1D grid of n_factors to search across
	tol : float, the early-stopping tolerance for the NMF model
	b_size : int, the model batch size
	n_epochs : int, the maximum number of model training epochs
	lr : float, the model's Adam optimizer initial learning rate

	Returns
	--------
	averages_df : pd.DataFrame, tabular output of cross validation
	"""
	# get indices, for k-folds cross validation
	k_fold_indices = ms_imputer.utilities.k_folds_split(
													matrix, 
													folds
	)
	results = []

	# k-folds cross validation loop
	#for k in range(0, folds):
	for k in tqdm(range(0, folds), unit="fold"):
		# partition
		train, val = ms_imputer.utilities.kfolds_partition(
											matrix, 
											k_fold_indices, 
											k,
		)

		# hyperparameter search
		for n_factors in grid:
			# init model
			nmf_model = GradNMFImputer(
							n_rows=train.shape[0], 
							n_cols=train.shape[1], 
							n_factors=n_factors, 
							stopping_tol=tol, 
							train_batch_size=b_size, 
							eval_batch_size=b_size,
							n_epochs=n_epochs, 
							loss_func="MSE",
							optimizer=torch.optim.Adam,
							optimizer_kwargs={"lr": lr},
			)
			# fit model, get reconstruction
			recon = nmf_model.fit_transform(train, val)

			# collect errors
			train_loss = ms_imputer.utilities.mse_func_np(train, recon)
			val_loss = ms_imputer.utilities.mse_func_np(val, recon)

			res = {
				"fold": k,
				"n_factors": n_factors,
				"train_error": train_loss,
				"valid_error": val_loss,
			}

			results += [pd.DataFrame(res, index=[0])]
    
	# format nested cross validation results dataframe
	results = pd.concat(results)
	results = results.reset_index(drop=True)

	averages_df = pd.DataFrame(columns=["n_factors", \
								"train_error", "valid_error"])
	# average results across each of k folds
	for factor in grid:
		sub_df = results[results["n_factors"] == factor]
		train_err_mean = np.mean(sub_df["train_error"])
		valid_err_mean = np.mean(sub_df["valid_error"])

		averages_df = averages_df.append({
	    							'n_factors' : factor, 
	    							'train_error' : train_err_mean, 
	    							'valid_error' : valid_err_mean},
	    							ignore_index=True
		)

	return averages_df

@click.command()
@click.option("--csv_path", type=str, 
				help="path to the input matrix (.csv)",
				required=True)
@click.option("--output_stem", type=str,
				help="file stem to use for output file",
				required=True)
@click.option("--learning_rate", type=float,
				help="the optimizer learning rate", 
				required=False, default=0.05)
@click.option("--max_epochs", type=int,
				help="max number of training epochs", 
				required=False, default=3000)	
@click.option("--k_folds", type=int,
				help="number of cross validation folds", 
				required=False, default=3)		
@click.option("--min_present", type=int,
				help="minimum number of present values for a peptide or protein", 
				required=False, default=4)	

def main(
		csv_path, 
		output_stem,
		learning_rate, 
		max_epochs,
		k_folds,
		min_present,
):
	"""  
	Search across a 1D grid of hyperparameters to select the optimal
	value for n_factors. Then train an NMF model with the optimal
	value of n_factors, and reconstruct missing values in the input
	matrix. 

	Arguments
	---------
	csv_path : str, posix.Path to the input {peptide, protein} quants
				matrix. Can be trimmed or raw MaxQuant output. 
				Required. 
	output_stem : str, file stem to use for the reconstructed matrix
				output file. 
				Required.
	learning_rate : float, the initial learning rate to use for the 
				model's (Adam) optimizer.
				Optional. Default=0.05.
	max_epochs : int, the maximum number of training epochs for the model.
				Optional. Default=3000.
	k_folds : int, k, for k-folds cross validation. 
				Optional. Default=10.
	min_present : int, the minimum number of present values for a peptide
				or protein required for the model to impute that peptide
				or protein. 
				Optional. Default=5.
	"""
	# Default model configs. Not overwritable
	tolerance = 0.0001
	batch_size = 1000
	factors_grid = [1,2,4,8,16,32]

	# trim the input file, if need be
	trim_bool = ms_imputer.utilities.maxquant_trim(csv_path, output_stem)
	if trim_bool:
		quants_path = output_stem + "_quants.csv"
	else:
		quants_path = csv_path

	# read in quants matrix, replace 0s with nans
	quants_matrix = pd.read_csv(quants_path)
	quants_matrix.replace([0, 0.0], np.nan, inplace=True)
	quants_matrix = np.array(quants_matrix)

	# discard (mostly) empty rows
	quants_matrix_trim, discard_idx = ms_imputer.utilities.\
												discard_empty_rows(
													quants_matrix, 
													min_present
	)
	print("running cross validation")
	# run cross validation hyperparameter search
	cross_valid_res = cross_validate(
								quants_matrix_trim, 
								k_folds, 
								factors_grid,
								tolerance, 
								batch_size, 
								max_epochs, 
								learning_rate,
	)

	# select the optimal choice in latent factors
	val_err_min = np.min(cross_valid_res["valid_error"])
	min_fold_df = cross_valid_res[cross_valid_res["valid_error"] == val_err_min]
	optimal_factors = list(min_fold_df["n_factors"])[0]

	print("training model with optimal choice in latent factors")

	# partition -- warning: this will exclude additional rows
	# train, val = ms_imputer.utilities.partition(
	# 									quants_matrix_trim, 
	# 									val_frac=0.1, 
	# 									min_present=min_present,
	# )

	# init optimal model
	nmf_model_opt = GradNMFImputer(
						n_rows=quants_matrix_trim.shape[0], 
						n_cols=quants_matrix_trim.shape[1], 
						n_factors=int(optimal_factors), 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_epochs, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate},
	)
	# fit model, get reconstruction
	optimal_recon = nmf_model_opt.fit_transform(quants_matrix_trim)

	# write optimally reconstructed matrix to csv
	pd.DataFrame(optimal_recon).to_csv(
					output_stem + 
					"_reconstructed.csv",
					index=False
	)
	# cleanup
	try:
		os.remove(output_stem + "_quants.csv")
	except FileNotFoundError:
		pass

	print("Done!")

if __name__ == "__main__":
    main()

