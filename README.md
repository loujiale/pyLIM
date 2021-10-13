# pyLIM

A python-based linear inverse modeling suite.

**pyLIM** is based on the linear inverse model (LIM) described by Penland & Sardeshmukh (1995).
This package provides the machinery to both calibrate and forecast/integrate a LIM.

## Installation
pyLIM requires Python 3.7+

pyLIM can be installed by cloning the GitHub repository:

```sh
git clone https://github.com/splillo/pyLIM
cd pyLIM
python setup.py install
```

## Dependencies
- cartopy >= 0.17.0
- matplotlib >= 2.2.2
- numpy >= 1.14.3
- scipy >= 1.1.0
- xarray >= 0.8.0
- global-land-mask >= 1.0.0

Create and activate python environment with dependencies. Called limenv.
```sh
cd pyLIM
conda env create -f environment.yml
conda activate limenv
```

