Usage
=======

`$ ms_imputer --help` should return usage information

```
Usage: ms_imputer [OPTIONS]

  Fit an NMF model to the input matrix, impute missing values.

Options:
  --csv_path TEXT        path to the input matrix (.csv)  [required]
  --output_stem TEXT     file stem to use for output file  [required]
  --output_path TEXT     path to output file
  --factors INTEGER      number of factors to use for reconstruction
  --learning_rate FLOAT  the optimizer learning rate
  --max_epochs INTEGER   max number of training epochs
  --help                 Show this message and exit.
```
