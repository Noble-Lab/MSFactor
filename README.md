ms_imputer
==========
[![PyPi](https://img.shields.io/pypi/v/ms_imputer.svg)](https://pypi.python.org/pypi/ms_imputer)
[![Build Status](https://img.shields.io/travis/lincoln-harris/ms_imputer.svg)](https://travis-ci.com/lincoln-harris/ms_imputer)     

What is `ms_imputer`?
--------------------

This tool uses non-negative matrix factorization to impute missing values in quantitative mass spectrometry data. 

Installation
------------

With the python standard library [`venv`](https://docs.python.org/3/library/venv.html) module
```
python3 -m venv ms_imputer
source ms_imputer/bin/activate
pip3 install -e . 
```

With [conda](https://docs.conda.io/en/latest/)
```
conda create -n ms_imputer python=3.7
conda activate ms_imputer
pip3 install -e . 
```

Usage
-----
```
Usage: ms_imputer [OPTIONS]

  Fit an NMF model to the input matrix, impute missing values.

Options:
  --csv_path TEXT        path to the input matrix (.csv) [required]
  --output_stem TEXT     file stem to use for output file [required]
  --factors INTEGER      number of factors to use for reconstruction
  --learning_rate FLOAT  the optimizer learning rate
  --max_epochs INTEGER   max number of training epochs
  --help                 Show this message and exit.
```    

Authors
-------

This work was produced by [Lincoln Harris](https://github.com/lincoln-harris) and [Bill Noble](https://github.com/wsnoble), of the University of Washington, and [Will Fondrie](https://github.com/wfondrie) of [Talus Bioscience](https://www.talus.bio/). For questions please contact lincolnh@uw.edu. 

Contributing
------------

We welcome any bug reports, feature requests or other contributions. 
Please submit a well documented report on our [issue tracker](https://github.com/lincoln-harris/ms_imputer/issues). 
For substantial changes please fork this repo and submit a pull request for review. 

See [CONTRIBUTING.md](https://github.com/lincoln-harris/ms_imputer/blob/main/docs/CONTRIBUTING.md) for additional details. 

You can find official releases [here](https://github.com/lincoln-harris/ms_imputer/releases). 


