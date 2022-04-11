""" 
The commandline entry point for ms_imputer
"""
import pandas as pd 
import numpy as np
import click
import torch
import os

from ms_imputer.models.linear import GradNMFImputer
import ms_imputer.utilities

@click.command()
@click.option("--csv_path", type=str, 
				help="path to the input matrix (.csv)",
				required=True)
@click.option("--output_stem", type=str,
				help="file stem to use for output file",
				required=True)
@click.option("--learning_rate", type=float,
				help="the optimizer learning rate", 
				required=False)
@click.option("--max_epochs", type=int,
				help="max number of training epochs", 
				required=False)
def main(
		csv_path, 
		output_stem,
		learning_rate=None, 
		max_epochs=None,
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
				Optional. Default=0.0001.
	max_epochs : int, the maximum number of training epochs for the model.
				Optional. Default=3000.
	"""
	# Default model configs. These will be overwritten by cmdline args
	_learning_rate = 0.05
	_max_epochs = 3000

	# Default model configs. Not overwritable
	tolerance = 0.0001
	batch_size = 1000
	factors_grid = [1,2,4,8,16,32]
	#factors_grid = [1,2,4]
	k_folds = len(factors_grid)

	# for selecting the k-index of the validation split
	k_remaining = list(range(0, k_folds)) 

	# Overwrite default configs, if specified at cmdline
	if learning_rate:
		_learning_rate = learning_rate
	if max_epochs:
		_max_epochs = max_epochs

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

	# get indices, for k-folds cross validation
	k_fold_indices = ms_imputer.utilities.shuffle_and_split(quants_matrix, k_folds)
	cross_valid_results = []

	# k-folds cross validation loop. Implements hyperparameter search
	for k in range(0, k_folds):
		print("working on fold: ", k)
		train, val, test, k_remaining = ms_imputer.utilities.get_kfold_sets(
											quants_matrix, k_fold_indices, 
											k, k_remaining)
		n_factors = factors_grid[k]

		# init model
		nmf_model = GradNMFImputer(
						n_rows=train.shape[0], 
						n_cols=train.shape[1], 
						n_factors=n_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=_max_epochs, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": _learning_rate}
					)
		# fit model, get reconstruction
		recon = nmf_model.fit_transform(train, val)

		# collect errors
		train_loss = ms_imputer.utilities.mse_func_np(train, recon)
		val_loss = ms_imputer.utilities.mse_func_np(val, recon)
		test_loss = ms_imputer.utilities.mse_func_np(test, recon)

		res = {
			"fold": k,
			"n_factors": n_factors,
			"train_error": train_loss,
			"valid_error": val_loss,
			"test_error": test_loss,
		}

		cross_valid_results += [pd.DataFrame(res, index=[0])]
    
	# format cross validation results dataframe
	cross_valid_results = pd.concat(cross_valid_results)
	cross_valid_results = cross_valid_results.reset_index(drop=True)
	cross_valid_results.to_csv("cross_valid_results.csv", index=False)

	# select the optimal choice in latent factors
	val_err_min = np.min(cross_valid_results["valid_error"])
	min_fold_df = cross_valid_results[cross_valid_results["valid_error"] == val_err_min]
	optimal_factors = list(min_fold_df["n_factors"])[0]

	# partition
	train, val, test = ms_imputer.utilities.split(
							quants_matrix, val_frac=0.1, test_frac=0.01, 
							min_present=0)

	print("training model with optimal choice in latent factors")
	# init optimal model
	nmf_model_opt = GradNMFImputer(
						n_rows=train.shape[0], 
						n_cols=train.shape[1], 
						n_factors=optimal_factors, 
						stopping_tol=tolerance, 
						train_batch_size=batch_size, 
						eval_batch_size=batch_size,
						n_epochs=_max_epochs, 
						loss_func="MSE",
						optimizer=torch.optim.Adam,
						optimizer_kwargs={"lr": _learning_rate}
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

