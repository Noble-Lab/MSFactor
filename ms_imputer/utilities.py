"""
UTILITIES

ms_imputer utilities
"""
import pandas as pd
import numpy as np

def maxquant_trim(maxquant_path, out_stem):
	"""
	Keep only the peptide/protein "Intensity" columns from 
	Maxquant's proteinGroups.txt and peptides.txt standard
	output file formats. Writes new file to specified path

	Parameters
	----------
	maxquant_path : str, posix.path to maxquant proteinGroups.txt
					or peptides.txt output file
	out_stem : str, output file stem
	out_path : str, posix.path, where to write the output file
	
	Returns
	-------
	none
	"""
	raw_maxquant = pd.read_csv(maxquant_path, sep="\t", dtype=object)

	# identify just the protein/peptide quant columns
	intensity_cols = []
	for x in list(raw_maxquant.columns):
	    if 'Intensity' in x:
	        intensity_cols.append(x)

	# subset
	df_trim = raw_maxquant[intensity_cols]

	# drop these summary cols
	if 'Intensity' in df_trim.columns:
	    df_trim = df_trim.drop(['Intensity'], axis=1)
	if 'Intensity L' in df_trim.columns:
	    df_trim = df_trim.drop(['Intensity L'], axis=1)
	if 'Intensity H' in df_trim.columns:
	    df_trim = df_trim.drop(['Intensity H'], axis=1)

	# write to csv
	df_trim.to_csv(out_stem + "_quants.csv", index=None)


