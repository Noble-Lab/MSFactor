"""
TEST UTILITIES

Testing the ms_imputer module utility functions
"""
import numpy as np
import pandas as pd
import unittest
import os
import pytest

from ms_imputer.utilities import maxquant_trim

class UtilitiesTester(unittest.TestCase):
	def test_maxquant_trim(self):
		"""
		Test the utilities.maxquant_trim function, with
		an expected trimmed matrix 
		"""
		raw_mq = "maxquant_raw.csv"
		output_stem = "mq-tester"
		maxquant_trim(raw_mq, output_stem)

		assert os.path.exists(output_stem + "_quants.csv")

		mq_func_trim = np.array(pd.read_csv(output_stem + "_quants.csv"))
		truth_trim = np.array(pd.read_csv("maxquant_trimmed.csv"))

		assert np.array_equal(mq_func_trim, truth_trim)

		os.remove(output_stem + "_quants.csv")



