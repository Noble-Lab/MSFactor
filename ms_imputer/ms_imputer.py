""" 
The commandline entry point for ms_imputer
"""
import pandas as pd 
import numpy as np
import click
import torch

from ms_imputer.models.linear import GradNMFImputer

@click.command()
@click.option("--csv_path", type=str, 
				help="path to the trimmed input file")
@click.option("--pxd", type=str,
				help="protein exchange identifier")
@click.option("--output_path", type=str,
				help="path to output file")
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
		pxd,
		output_path, 
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
	
	# read in quants matrix, replace 0s with nans
	quants_matrix = pd.read_csv(csv_path)
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

	# write reconstructed matrix to csv
	pd.DataFrame(recon).to_csv(
					output_path + pxd + "_nmf_reconstructed.csv", 
					index=False
	)

	print("Done!")
	print(" ")

if __name__ == "__main__":
    main()

