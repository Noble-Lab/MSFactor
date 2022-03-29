""" 
FIT_NMF

This module fits an NMF model for a single PRIDE matrix. 
If requested, fits models across a range of latent factors.
Need to specify cmd line args.
"""
import pandas as pd 
import numpy as np
from argparse import ArgumentParser

from nmf_modeler import NMFModeler, Dataset

def parse_args():
	""" 
	Parse the CLI arguments 
	"""
	parser = ArgumentParser()
	parser.add_argument("--csv_path", type=str,
						help="path to the trimmed input file")
	parser.add_argument("--PXD", type=str,
						help="protein exchange identifier")
	parser.add_argument("--output_path", type=str,
						help="path to output file")
	parser.add_argument("--factors", type=int,
						help="number of factors to use for reconstruction",
						required=False)
	parser.add_argument("--learning_rate", type=float,
						help="the optimizer learning rate", 
						required=False)
	parser.add_argument("--max_epochs", type=int,
						help="max number of training epochs", 
						required=False)

	return parser.parse_args()

def main():
	"""  
	The main function. Gets commandline args, partitions the data, 
	runs the selected model across a range of latent factors, and stores 
	results to a csv, in the model-out/ directory. 
	"""
	# get CLI arguments
	args = parse_args()

	# set default configs
	if args.factors:
		n_factors = args.factors 
	else: 
		n_factors = 8

	if args.learning_rate:
		lr = args.learning_rate
	else:
		lr = 0.05

	if args.max_epochs:
		max_epochs = args.max_epochs
	else:
		max_epochs = 3000

	# get peptide quants dataset
	peptides_dataset = Dataset(pxd=args.PXD, data_pth=args.csv_path)

	# init model
	model = NMFModeler(max_epochs=max_epochs, learning_rate=lr)
	
	# fit model, get reconstruction
	model.fit(
			pxd=args.PXD,
			train=peptides_dataset.quants_matrix, 
			val=None,
			n_factors=n_factors
	)

	# write reconstructed matrix to csv
	pd.DataFrame(model.recon).to_csv(args.output_path + "nmf_reconstructed.csv")

if __name__ == "__main__":
	main()
