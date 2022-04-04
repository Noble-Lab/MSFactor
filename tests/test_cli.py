"""
TEST-CLI

Tests for the commandline interface for the 
ms-imputer module, i.e. ms_imputer.py
"""
import unittest
import pytest
import os
import pandas as pd 
import numpy as np

from click.testing import CliRunner

from ms_imputer.ms_imputer import main

# define configs
csv_tester = "peptide_quants_tester.csv"
output_stem = "tester"

class CliTester(unittest.TestCase):
	def test_cli_required(self):
		"""
		Make sure the cli is able to pare the required
		commandline arguments. 
		"""
		runner = CliRunner()
		result = runner.invoke(main, 
							["--csv_path", csv_tester, 
						 	"--output_stem", output_stem])
		
		assert result.exit_code == 0
		assert os.path.exists(output_stem + "_reconstructed.csv")
		
		os.remove(output_stem + "_reconstructed.csv")

	def test_cli_adnl(self):
		"""
		Make sure cli is able to parse the optional cmdline arguments
		"""
		runner = CliRunner()
		result = runner.invoke(main, 
							["--csv_path", csv_tester, 
						 	"--output_stem", output_stem, 
						 	"--factors", 4, 
						 	"--learning_rate", 0.1,
						 	"--max_epochs", 10])
		
		assert result.exit_code == 0
		assert os.path.exists(output_stem + "_reconstructed.csv")

		os.remove(output_stem + "_reconstructed.csv")


