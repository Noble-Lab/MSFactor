""" 
MS_IMPUTER

This module fits an NMF model for a single PRIDE matrix. 
If requested, fits models across a range of latent factors.
Need to specify cmd line args.
"""
import pandas as pd 
import numpy as np
import click
import torch

from ms_imputer.models.linear import GradNMFImputer

@click.command()
@click.option("--csv_path", type=str, 
				help="path to the trimmed input file")
@click.option("--PXD", type=str,
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
		PXD,
		output_path, 
		factors=None, 
		learning_rate=None, 
		max_epochs=None
):
	"""  
	Fit an NMF model to the input matrix, impute missing values.
	"""

	# set default configs
	if factors:
		n_factors = factors 
	else: 
		n_factors = 8

	if learning_rate:
		lr = learning_rate
	else:
		lr = 0.05

	if max_iters:
		max_iters = max_epochs
	else:
		max_iters = 3000
	
	# read in quants matrix, replace 0s with nans
	quants_matrix = pd.read_csv(csv_path)
	quants_matrix.replace(0, np.nan, inplace=True)
	quants_matrix = np.array(quants_matrix)

	# init model
	nmf_model = GradNMFImputer(
					quants_matrix.shape[0], 
					quants_matrix.shape[1], 
					n_factors=n_factors, 
					stopping_tol=0.0001, 
					train_batch_size=1000, 
					eval_batch_size=1000,
					n_epochs=max_iters, 
					loss_func="MSE",
					optimizer=torch.optim.Adam,
					optimizer_kwargs={"lr": lr}
				)
	
	# fit model, get reconstruction
	recon = nmf_model.fit_transform(train, val)

	# write reconstructed matrix to csv
	pd.DataFrame(recon).to_csv(output_path + "nmf_reconstructed.csv")

if __name__ == "__main__":
    main()

