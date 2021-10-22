import ROOT
from array import array
import sys
import pathlib

sys.path.insert(0, "..")

from model_train import modelParameters
import pandas as pd
import numpy as np

from make_df import make_df

pathlib.Path('svb_fit').mkdir(parents=True, exist_ok=True)

sig_file = ROOT.TFile("../../../events/MG3/TTree/HH4b_dR04_toyTree.root")
sig_file.ls()
sig_tree = sig_file.Get("Tree")
dfS = make_df(sig_tree, fromnp=False, fourTag=True)
dfS.to_hdf("../../../events/MG3/dataframes/signal.h5", key="df", mode="w")

dfS['signal'] = 1

dfB = pd.read_hdf("../../../events/MG3/dataframes/bbbb_large.h5")
dfB = dfB[dfB.CR|dfB.SR]
dfS = dfS[dfS.CR|dfS.SR]
dfB['signal'] = 0

model = modelParameters(dfB, dfS, model_path="svb_fit/", classifier="SvB")
model.trainSetup()
model.runEpochs(print_all_epochs=True)

