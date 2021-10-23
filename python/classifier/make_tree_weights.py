import ROOT
from array import array
import sys
import pathlib
from model_train import modelParameters
import pandas as pd
import numpy as np

from make_df import make_df


import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data', default='MG3', type=str, help='Nickname for dataset to use.')
parser.add_argument('-c', '--classifier', default='SvB', type=str, help='Name of classifier to use (SvB or FvT).')
args   = parser.parse_args()

data = args.data
class_type = args.classifier

if class_type == "SvB":
    classifier_path = "SvB/svb_fit/SvB_ResNet_6_6_6_np799_lr0.008_epochs10_stdscale_epoch10_loss0.2518.pkl"

else:
    classifier_path = "FvT/fvt_fit/FvT_ResNet_6_6_6_np799_lr0.01_epochs10_stdscale_epoch10_loss0.6823.pkl"

### Create Dirs
pathlib.Path('temp_trees').mkdir(parents=True, exist_ok=True)
pathlib.Path('temp_trees/SvB').mkdir(parents=True, exist_ok=True)
pathlib.Path('temp_trees/FvT').mkdir(parents=True, exist_ok=True)

### Load h5 Dataframes
df4bLarge = pd.read_hdf("../../events/MG3/dataframes/bbbb_large.h5")
df4bLarge = df4bLarge[df4bLarge.CR|df4bLarge.SR]
df4bLarge["signal"] = 0

df3b = pd.read_hdf("../../events/MG3/dataframes/bbbj.h5")
df3b = df3b[df3b.CR|df3b.SR]
df3b["signal"] = 0

df4b = pd.read_hdf("../../events/MG3/dataframes/bbbb.h5")
df4b = df4b[df4b.CR|df4b.SR]
df4b["signal"] = 0

dfS = pd.read_hdf("../../events/MG3/dataframes/signal.h5")
dfS = dfS[dfS.SR|dfS.CR]
dfS["signal"] = 1

### Load ROOT TFiles
bbbb_file       = ROOT.TFile("../../events/" + data + "/TTree/bbbb.root",              "READ")
bbbj_file       = ROOT.TFile("../../events/" + data + "/TTree/bbbj.root",              "READ")
bbbb_large_file = ROOT.TFile("../../events/" + data + "/TTree/bbbb_large.root",        "READ")
sig_file        = ROOT.TFile("../../events/" + data + "/TTree/HH4b.root", "READ")

### ROOT ROOT TTrees
bbbb_tree       = bbbb_file.Get("Tree")
bbbj_tree       = bbbj_file.Get("Tree")
bbbb_large_tree = bbbb_large_file.Get("Tree")
sig_tree        = sig_file.Get("Tree")

tree_names = ["bbbb", "bbbb_large", "bbbj", "HH4b_dR04_toyTree"]
trees      = [bbbb_tree, bbbb_large_tree, bbbj_tree, sig_tree]
dataframes = [df4b, df4bLarge, df3b, dfS]

model = modelParameters(df4bLarge, dfS, classifier=class_type, fileName=classifier_path)

for k in range(len(trees)):
    print("===============================    ", k, "        ==================================")
    tree = trees[k]

    #weights = [np.load("../results/" + data + "/plot_trees/svb_weights/" + names[l] + "_" + weight_names[k] + ".npy") for l in range(3)]
    h1, _ = model.predict(dataframes[k])
    h = h1[:,1]

    out_file = ROOT.TFile("temp_trees/" + class_type + "/" + tree_names[k] + ".root", "RECREATE")
    tree_new = tree.CloneTree(0)

    tree_new.Branch(class_type, np.array([0], dtype=np.float32), class_type + "/F")

    w = array('f',[0])
    tree_new.SetBranchAddress(class_type, w)

    o = 0

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.SR == 1 or tree.CR == 1:
            w[0] = h[o]
            tree_new.Fill()
            o+=1

    out_file.Write()
    out_file.Close()

bbbb_file.Close()
bbbj_file.Close()
bbbb_large_file.Close()
sig_file.Close()
   

for k in range(len(tree_names)):
    temp_file = ROOT.TFile("temp_trees/" + class_type + "/" + tree_names[k] + ".root", "READ") 
    temp_tree = temp_file.Get("Tree")

    new_file  = ROOT.TFile("../../events/" + data + "/TTree/" + tree_names[k] + ".root", "RECREATE")
    new_tree = temp_tree.CloneTree()

    new_file.Write()
    new_file.Close()

    temp_file.Close()
