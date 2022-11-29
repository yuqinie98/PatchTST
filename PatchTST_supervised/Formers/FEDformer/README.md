# FEDformer
From https://github.com/MAZiqing/FEDformer


Frequency Enhanced Decomposed
Transformer (FEDformer) is more efficient than
standard Transformer with a linear complexity
to the sequence length. 

Our empirical studies
with six benchmark datasets show that compared
with state-of-the-art methods, FEDformer can
reduce prediction error by 14.8% and 22.6%
for multivariate and univariate time series,
respectively.


## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from xxxx.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/run_M.sh
bash ./scripts/run_S.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
xxxxx
```

## Contact

If you have any question or want to use the code, please contact xxx@xxxx .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

