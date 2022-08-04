# HH4bsim
Data-Driven Background Modeling for Double Higgs Boson Production in the 4b Final State.
 
![SvB](/SvB_logy.pdf)




## Requirements
Python 3.5, ROOT, `pyROOT`, `numpy`, `pandas`, `sklearn`, `pytorch`, `scipy`,
            [`pot`](https://pythonot.github.io/) , [`energyflow`](https://energyflow.network/),
           [`PlotTools`](https://github.com/patrickbryant/PlotTools).

## Overview 

This repository is based on the methodologies developed in the preprint

[1] Manole, T., Bryant, P., Alison, J., Kuusela, M., Wasserman, L. (2022). Background Modeling for di-Higgs Boson Production: Density Ratios and Optimal Transport.

The following three background modeling methods are currently implemented.
* `FvT`: Train the FvT (Four vs. Three) classifier to discriminate 4b from 3b events in a Control Region, and extrapolate to the Signal Region.
* `OT-kNN`: Estimate an optimal transport coupling between the Control and Signal Regions of the 3b events, and extrapolate to 4b events using nearest neighbor methods.
* `OT-FvT`: Estimate an optimal transport coupling between the Control and Signal Regions of the 3b events, and extrapolate to 4b events using the FvT classifier in the Control Region

This repository also contains a simulated dataset of events, which is described in Section 6 of [1], and was generated using 
[this script](https://github.com/patrickbryant/ZZ4b/blob/e34f45da0def11736460ec4a503a961d6cb3781d/python/analysis.py). 

## Getting Started

To set up the repository for use, run the `preprocess_dataset.sh` script:
```
cd python/event_scripts
source preprocess_dataset.sh
```

The main script for fitting and plotting background models is `background_analysis.py`. In order to reproduce the plots and validation results in Section 6 of [1], run the following commands. 

```
cd python
python background_analysis.py -m method_name -p True -rh True -v True
```
Here, `-m` refers to the method name, which can be one of `HH-FvT`, `HH-Comb-FvT`, `HH-OT`, or `benchmark` which simply refers to a rescaling of the 3b dataset.
The `-p` switch indicates that histograms of the fit should produced. These plots require `.hist` ROOT files which are reproduced whenever the `-rh` switch is specified. The `-v` switch calls the FvT classifier to be trained between the fit and true 4b data as a proxy for their discrepancy. This leads to Figure X of paper [1]. To produce plots which summarize the three methods, run
```
python background_analysis.py -sp True
```

To refit any of the methods, run
``` 
python background_analysis.py -m method_name -f True  
```
Note that for the OT-kNN and OT-FvT methods, the above command will require couplings between the 3b Control and Signal Region distributions. These large files are available upon request, or can be reproduced using the script `python/make_large_coupling.py`. For the FvT method, the FvT classifier can be retrained using the following commands:

```
cd python/classifier/FvT
python train.py
cd ..
python make_tree_weights.py -c FvT
```
