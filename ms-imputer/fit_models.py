""" 
FIT_MODELS

This module fits a model from: {NMF, KNN, MissForest, Perceptron},
for a single PRIDE matrix. Fits models across a range 
of latent factors, for NMF and KNN. Need to specify cmd line args.
"""
import sys
sys.path.append('../../../bin/')

import pandas as pd 
import numpy as np
import yaml
from argparse import ArgumentParser

from modeler import Modeler, Dataset

def parse_args():
	""" 
	Parse the CLI arguments 
	"""

	parser = ArgumentParser()
	parser.add_argument("--csv_path", type=str,
						help="path to the trimmed input file")
	parser.add_argument("--PXD", type=str,
						help="protein exchange identifier")
	parser.add_argument("--model", type=str,
						help="options: NMF, KNN, MissForest")
	parser.add_argument("--config_path", type=str,
						help="path to config.yml")
	parser.add_argument("--output_tail", type=str, required=False,
						help="tail (name) for the output file")

	return parser.parse_args()

def main():
	"""  
	The main function. Gets commandline args, partitions the data, 
	runs the selected model across a range of latent factors, and stores 
	results to a csv, in the model-out/ directory. 
	"""
	# get CLI arguments
	args = parse_args()

	# open the config file
	with open(args.config_path) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	# get hyperparameters
	if args.model == "NMF" or args.model == "KNN":
		hparams_range = config["factors_range"]
	elif args.model == "MissForest":
		hparams_range = config["trees_range"]
	else:
		raise ValueError("Unrecognized model type.")

	# get peptide quants dataset
	peptides_dataset = Dataset(pxd=args.PXD, data_pth=args.csv_path)

	# log transform
	if config["log_transform"]:
		peptides_dataset.log_transform()

	# partition
	peptides_dataset.partition()

	results = []

	# hyperparameter search
	for hparam in hparams_range:
		model = Modeler(
					model_type=args.model, 
					loss_func=config["loss_func"],
					log_transform=config["log_transform"],
					tolerance=config["tolerance"],
					max_epochs=config["max_iters"],
					batch_size=config["batch_size"],
					learning_rate=config["learning_rate"],
					parallelize=config["parallelize"],
				)
		# fit model, get reconstruction
		recon = model.fit(
					peptides_dataset.train, 
					peptides_dataset.val,
					n_factors=hparam, 
					pxd=args.PXD
				)
		# get errors
		train_res, val_res, test_res = model.collect_errors(
										peptides_dataset.train, 
										peptides_dataset.val, 
										peptides_dataset.test, 
										recon, 
										hparam
										)
		# get intermediate plots
		if model.type == "NMF":
			model.get_loss_curves(n_factors=hparam, pxd=args.PXD, 
									tail=args.output_tail)

		if model.type == "NMF" or model.type == "MissForest":
			model.get_correlation_plots(
							peptides_dataset.val, 
							n_factors=hparam, 
							pxd=args.PXD,
							tail=args.output_tail
					)

		results += [pd.DataFrame(train_res, index=[0]), 
					pd.DataFrame(val_res, index=[0]),
					pd.DataFrame(test_res, index=[0])]

	# summarize results
	results = pd.concat(results)
	results = results.reset_index(drop=True)

	if args.output_tail:
		output_tail = args.output_tail
	else:
		output_tail = ""

	# write results
	results.to_csv("model-out/" + args.PXD + "." + args.model + 
					"_" + output_tail + ".csv", index=False)

if __name__ == "__main__":
	main()
