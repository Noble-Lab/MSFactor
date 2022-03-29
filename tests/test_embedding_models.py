""" tests for bin/embeddingModels.py """
import sys
import unittest
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.append('../bin')
sys.path.append('bin/')
from util_functions import *

class EmbeddingModelsTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		""" __init__ method for class object """
		self.foobar = 0
		self.helloworld = 1

	def test_always_passes(self):
		""" dummy test """
		self.assertTrue(True)
