"""
GENERATE-PLOTS

This module generates output plots for the nmf-modeler class. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import seaborn as sns

# plotting templates
sns.set(context="talk", style="ticks") 
pal = sns.color_palette()

def plot_train_loss(model, PXD, n_row_factors, n_col_factors, 
						model_type, eval_loss="MSE", tail=None):
	""" 
	Generate training loss vs training iter plot for
	NMF or Perceptron, using the history parameter. 
	For training and validation sets. 

	Parameters
	----------
	model : Modeler, the fitted NMF model
	PXD : str, the proteome exchange identifier
	n_row_factors : int, number of row factors, or, number of factors
	n_col_factors : int, number of col factors
	model_type : str, {"NMF", "Perceptron", "DNN"}
	eval_loss : str, eval loss function {"MSE", "RMSE"}
	tail : str, the config file to associate this plot with. 
				OPTIONAL
	Returns
	-------
	none
	"""
	plt.figure()
	plt.plot(list(model.history.epoch[2:]), 
		list(model.history["Train MSE"][2:]), 
		label="Training loss")
	plt.plot(list(model.history.epoch[2:]), 
		list(model.history["Validation MSE"][2:]), 
		label="Validation loss")
	plt.legend()
	plt.xlabel("epochs")
	plt.ylabel(eval_loss)

	if not tail:
		tail = ""

	if model_type == "Perceptron" or model_type == "DNN":
		plt.savefig("logs/" + model_type + "-training-errors-" + 
					PXD + "_" + str(n_row_factors) + "_" + str(n_col_factors) + 
					"_factors" + "_" + tail + ".png", 
					dpi=250, bbox_inches='tight')
	if model_type == 'NMF':
		plt.savefig("logs/" + model_type + "-training-errors-" + 
					PXD + "_" + str(n_row_factors) +  
					"_factors" + "_" + tail + ".png", 
					dpi=250, bbox_inches='tight')

	plt.close()
	return

def real_v_imputed(recon_mat, val_mat, PXD, row_factors, 
							col_factors, model_type, 
							log_transform=False, tail=None):
	""" 
	Generate an real vs imputed abundance plot for a given 
	model. For validation set. 

	Parameters
	----------
	recon_mat : np.ndarray, the reconstructed matrix
	val_mat : np.ndarray, the validation matrix
	PXD: str, the PRIDE identifier
	row_factors : int, the number of row factors used for reconstruction
	col_factors : int, the number of column factors used for reconstruction
	model_type : str, {"DNN", "Perceptron", "NMF", "MissForest"}
	log_transform : boolean, have the data been logged or not? 
	tail : str, the config file to associate this plot with. 
				OPTIONAL
	Returns
	-------
	None
	"""
	# get index of nan values in validation set
	val_nans = np.isnan(val_mat)

	# get non nan values in each of val and recon matrices
	val_set = val_mat[~val_nans]
	val_set_recon = recon_mat[~val_nans]

	# get Pearson's correlation coefficient
	cor_mat = np.corrcoef(x=val_set, y=val_set_recon)
	pearson_r = np.around(cor_mat[0][1], 2)

	# initialize the figure
	plt.figure(figsize=(8,6))
	ax = sns.scatterplot(x=val_set, y=val_set_recon, alpha=0.3)
	
	ax.set_xlabel('real abundance')
	ax.set_ylabel('imputed abundance')

	if not log_transform:
		plt.xscale("log")
		plt.yscale("log")

	# add the correlation coefficient
	ax.text(0.95, 0.05, "R: %s"%(pearson_r),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)

	set_min = min(np.min(val_set), np.min(val_set_recon))
	set_max = max(np.max(val_set), np.max(val_set_recon))

	if set_min < 1:
		set_min = 1

	#ax.set_xlim(left=set_min, right=set_max)
	#ax.set_ylim(bottom=set_min, top=set_max)

	# add diagonal line
	x = np.linspace(set_min, set_max, 100)
	y = x
	plt.plot(x, y, '-r', label='y=x', alpha=0.6)

	plt.minorticks_off()

	if not tail:
		tail = ""

	# write figure to file
	if model_type == "Perceptron" or model_type == "DNN":
		plt.savefig("logs/" + model_type + "-real-v-imputed-" + 
					PXD + "_" + str(row_factors) + "_" + str(col_factors) + 
					"_factors" + "_" + tail + ".png", 
					dpi=250, bbox_inches='tight')
	if model_type == 'NMF' or model_type == "MissForest":
		plt.savefig("logs/" + model_type + "-real-v-imputed-" + 
					PXD + "_" + str(row_factors) +  
					"_factors" + "_" + tail + ".png", 
					dpi=250, bbox_inches='tight')

	plt.close()
	return

def aggregate_results(projects, trimmed_dir):
	""" 
	Aggregates the results into a single dataframe. 
	Writes output to a csv
	
	Parameters
	----------
	projects : list, all of the Protein Exchange Identifiers
	trimmed_dir : PosixPath, path to the trimmed peptides.txt files

	Returns
	-------
	None
	"""

	def get_mv_stats(project):
		""" 
		Returns the MV fraction and number of present values in a 
		(trimmed) PRIDE matrix
		
		Parameters
		----------
		project : str, single PRIDE identifier
		
		Returns
		-------
		present_values : int, the number of present values in the dataset
		mv_frac : float, the missingness fraction in the dataset
		"""
		df = pd.read_csv(trimmed_dir + project +
								 "_peptides.csv").to_numpy()
		pv_count = np.count_nonzero(df)
		pv_frac = np.count_nonzero(df) / df.size
    
		mv_frac = 1 - pv_frac

		return pv_count, mv_frac

	def get_factors_and_mse(df, nn=False):
		""" 
		Return the test set MSE associated with the optimal number
		of latent factors for the model, as well as the optimal
		number of latent factors. 

		Parameters
		----------
		df : pd.DataFrame, the "model-out/" dataframe. 

		Returns
		-------
		optimal_factors : int, the optimal number of latent factors 
		test_mse : float, the test set MSE associated with the 
					optimal number of latent factors for that model
		"""
		# find the index corresponding to the lowest validation error
		min_idx = df[df['split'] == 'Validation']['error'].idxmin()

		if not nn:
			# get the corresponding number of latent factors
			optimal_factors = df.iloc[min_idx]['n_factors']
			optimal_df = df[df['n_factors'] == optimal_factors]

		if nn:
			# get the corresponding number of latent factors
			row_factors = df.iloc[min_idx]['row_factors']
			col_factors = df.iloc[min_idx]['col_factors']
			optimal_factors = [row_factors, col_factors]

			optimal_df = df[(df['row_factors'] == row_factors) & 
								(df['col_factors'] == col_factors)]

		test_mse = list(optimal_df[optimal_df['split'] == 
								"Test"]["error"])[0]
		return optimal_factors, test_mse

	results = []

	for project in projects:
		nmf_df = pd.read_csv('model-out/' + project + ".NMF_.csv")
		knn_df = pd.read_csv('model-out/' + project + ".KNN_.csv")
		mf_df = pd.read_csv('model-out/' + project + ".MissForest_.csv")
		#percept_df = pd.read_csv('model-out/' + project + ".Perceptron.csv")
		#dnn_df = pd.read_csv('model-out/' + project + ".DNN.csv")

		nmf_factors, nmf_mse = get_factors_and_mse(nmf_df)
		knn_factors, knn_mse = get_factors_and_mse(knn_df)
		mf_factors, mf_mse = get_factors_and_mse(mf_df)
		#percept_factors, percept_mse = get_factors_and_mse(percept_df,
		#									nn=True)
		#dnn_factors, dnn_mse = get_factors_and_mse(dnn_df, 
		#									nn=True)

		pv_count, mv_frac = get_mv_stats(project)

		res = {
			"Project ID": project,
			"NMF_lf": nmf_factors,
			"NMF_test_MSE": nmf_mse,
			"KNN_lf": knn_factors,
			"KNN_test_MSE": knn_mse,
			"MissForest_lf": mf_factors,
			"MissForest_test_MSE": mf_mse, 
			#"Perceptron_row_factors": percept_factors[0],
			#"Perceptron_col_factors": percept_factors[1],
			#"Perceptron_test_MSE": percept_mse,
			#"DNN_row_factors": dnn_factors[0],
			#"DNN_col_factors": dnn_factors[1],
			#"DNN_test_MSE": dnn_mse,
			"Present values": pv_count,
			"Missingness fraction": mv_frac
		}
		results += [pd.DataFrame(res, index=[0])]

	results = pd.concat(results)
	results = results.reset_index(drop=True)
	results.to_csv("results/aggregated.csv", index=False)

	return

