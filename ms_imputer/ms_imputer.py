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
@click.option("--trim_input", type=bool,
				help="does the maxquant file need to be trimmed?",
				required=False)
@click.option("--factors", type=int,
				help="number of factors to use for reconstruction",
				required=False)
@click.option("--learning_rate", type=float,
				help="the optimizer learning rate", 
				required=False)
@click.option("--max_epochs", type=int,
				help="max number of training epochs", 
				required=False)
def main(
		csv_path, 
		output_stem,
		trim_input=True, 
		factors=None, 
		learning_rate=None, 
		max_epochs=None
):
	"""  
	Fit an NMF model to the input matrix, impute missing values.
	"""
	# Default model configs. Overwritten by commandline args
	n_factors = 8
	lr = 0.05
	max_iters = 3000
	tolerance = 0.0001
	batch_size = 1000

	# Overwrite default configs, if specified
	if factors:
		n_factors = factors 
	if learning_rate:
		lr = learning_rate
	if max_epochs:
		max_iters = max_epochs

	# trim the input file, if need be
	if trim_input:
		ms_imputer.utilities.maxquant_trim(csv_path, output_stem)
		quants_path = output_stem + "_quants.csv"
	else:
		quants_path = csv_path
	
	# read in quants matrix, replace 0s with nans
	quants_matrix = pd.read_csv(quants_path)
	quants_matrix.replace([0, 0.0], np.nan, inplace=True)
	quants_matrix = np.array(quants_matrix)

	# init model
	nmf_model = GradNMFImputer(
					n_rows=quants_matrix.shape[0], 
					n_cols=quants_matrix.shape[1], 
					n_factors=n_factors, 
					stopping_tol=tolerance, 
					train_batch_size=batch_size, 
					eval_batch_size=batch_size,
					n_epochs=max_iters, 
					loss_func="MSE",
					optimizer=torch.optim.Adam,
					optimizer_kwargs={"lr": lr}
				)

	# fit model, get reconstruction
	print(" ")
	print("fitting model")
	recon = nmf_model.fit_transform(quants_matrix)

	# write to csv
	pd.DataFrame(recon).to_csv(
					output_stem + 
					"_reconstructed.csv",
					index=False
	)

	if trim_input:
		os.remove(output_stem + "_quants.csv")

	print("Done!")
	print(" ")

if __name__ == "__main__":
    main()

