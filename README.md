# HH4bsim
Data-Driven Background Modeling for Double Higgs Boson Production in the 4b Final State.
 
<p align="center">
  <img src="http://stat.cmu.edu/~tmanole/SvB_logy.png" alt="SvB" style="width: 90%;align:center;"/>
</p>



## Requirements
Python 3.5, ROOT, `pyROOT`, `numpy`, `pandas`, `sklearn`, `pytorch`, `scipy`,
            [`pot`](https://pythonot.github.io/) , [`energyflow`](https://energyflow.network/),
           [`PlotTools`](https://github.com/patrickbryant/PlotTools).

## Overview 

This repository is based on the methodologies developed in the paper

[1] ...

The following three background modeling methods are currently implemented.
* `HH-FvT`: Train a classifier to discriminate 4b from 3b events (FvT, or Four-vs-Three) in a Control Region, and extrapolate to the Signal Region.
* `HH-OT`: Estimate an optimal transport coupling between the Control and Signal Regions of the 3b events, and extrapolate to 4b events using nearest neighbor methods.
* `HH-Comb`: Estimate an optimal transport coupling between the Control and Signal Regions of the 3b events, and extrapolate to 4b events using the FvT classifier in the Control Region

This repository also contains a simulated dataset of 3b and 4b events, described in Section X of [1]. 

## Getting Started

To set up the repository for use, run the `preprocess_dataset.sh` script:
```
cd python/event_scripts
source preprocess_dataset.sh
```

The main script for fitting and plotting background models is `background_analysis.py`. In order to reproduce the plots and validation results in Section X of [1], run the following commands. 

```
cd python
python background_analysis.py -m method_name -p True -rh True -v True
```
Here, `-m` refers to the dataset name, which can be one of `HH-FvT`, `HH-Comb-FvT`, `HH-OT`, or `benchmark` which simply refers to a rescaling of the 3b dataset.
The `-p` switch indicates that histograms of the fit should produced. These plots require `.hist` ROOT files which are reproduced whenever the `-rh` switch is specified. The `-v` switch indicates that a classifier should be trained between the fit and true 4b data as a proxy for their discrepancy. To produce plots which summarize the three methods, run
```
python background_analysis.py -sp True
```

To refit any of the methods, run
``` 
python background_analysis.py -m method_name -f True  
```
Note that for the HH-OT and HH-Comb methods, the above command will require coupling matrices between the 3b Control and Signal Regions. These large files are available upon request, or can be reproduced using the script `python/make_large_coupling.py`. For the HH-FvT method, the FvT classifier can be retrained using the following commands:

```
cd python/classifier/FvT
python train.py
cd ..
python make_tree_weights.py -c FvT
```
