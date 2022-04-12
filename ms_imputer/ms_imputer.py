""" 
MS-IMPUTER

The commandline entry point for ms_imputer.  
"""
import pandas as pd 
import numpy as np
import click
import torch
import os

from ms_imputer.models.linear import GradNMFImputer
import ms_imputer.utilities

def cross_validate(matrix, folds, grid, tol, b_size, n_epochs, lr):
	""" 
	Run cross validation, by training an NMF model for a given matrix
	for each of a set of hyperparameters (n_factors)
	
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
	results : pd.DataFrame, tabular output of cross validation
	"""
	# get indices, for k-folds cross validation
	k_fold_indices = ms_imputer.utilities.shuffle_and_split(matrix, folds)
	
	results = []

	# k-folds cross validation loop. Implements hyperparameter search
	for k in range(0, folds):
		print("working on fold: ", k)
		train, val = ms_imputer.utilities.get_kfold_sets(
													matrix, 
													k_fold_indices, 
													k,
		)
		n_factors = grid[k]

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
    
	# format cross validation results dataframe
	results = pd.concat(results)
	results = results.reset_index(drop=True)
	#results.to_csv("cross_valid_results.csv", index=False)

	return results

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
@click.option("--report_confidence", type=bool,
				help="report confidence intervals associated \
				with each prediction?", 
				required=False, default=False)			

def main(
		csv_path, 
		output_stem,
		learning_rate, 
		max_epochs,
		report_confidence,
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
	report_confidence : bool, report the confidence interval associated 
				with each prediction. Requires an additional round
				of cross validation.
				Optional. Default=False
	"""
	# Default model configs. Not overwritable
	tolerance = 0.0001
	batch_size = 1000
	factors_grid = [1,2,4,8,16,32]
	k_folds = len(factors_grid)

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

	# run cross validation hyperparameter search
	cross_valid_res = cross_validate(
								quants_matrix, 
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

	# partition -- don't withhold any rows
	train, val, test = ms_imputer.utilities.split(
										quants_matrix, 
										val_frac=0.1, 
										test_frac=0.01, 
										min_present=0,
	)

	print("training model with optimal choice in latent factors")
	# init optimal model
	nmf_model_opt = GradNMFImputer(
						n_rows=train.shape[0], 
						n_cols=train.shape[1], 
						n_factors=optimal_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=max_epochs, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": learning_rate},
	)
	# fit model, get reconstruction
	optimal_recon = nmf_model_opt.fit(train, val)
	optimal_recon = nmf_model_opt.transform(quants_matrix)

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
	print(" ")

if __name__ == "__main__":
    main()

