ms_imputer
==========
[![PyPi](https://img.shields.io/pypi/v/ms_imputer.svg)](https://pypi.python.org/pypi/ms_imputer)
[![Build Status](https://img.shields.io/travis/lincoln-harris/ms_imputer.svg)](https://travis-ci.com/lincoln-harris/ms_imputer)     

What is `ms_imputer`?
--------------------

This tool uses non-negative matrix factorization to impute missing values in quantitative mass spectrometry data. 

Installation
------------

* TODO

Usage
-----

```
$ python fit_nmf.py --csv_path /path/to/input --PXD str --output_path /path/to/output
```

Arguments
---------
`--csv_path` : _str_, path to input csv file (matrix with missing values). Required     
`--PXD` : _str_, protein exchange identifier. Required       
`--output_path` : _str_,  where to write output (reconstructed matrix without missing values). Required     
`--factors` : _int_, the number of latent factors to train NMF model with. Not required      
`--learning_rate` : _float_, the optimizer learning rate. Not required      
`--max_epochs` : _int_, max number of training epochs. Not required     

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


