import ROOT
import numpy as np
from array import array
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data',  default='MG3', type=str, help='Dataset name.')
args = parser.parse_args()


### Signal fraction
### Taken from Table 8.1 2016 HH data at https://cds.cern.ch/record/2644551?ln=en
SB_sig_fraction = 1.1/10420 
CR_sig_fraction = 1.8/7553
SR_sig_fraction = 3.8/7134

### Load trees

sig_file = ROOT.TFile("../../events/" + args.data + "/TTree/HH4b_dR04_toyTree.root", "READ")
out_file = ROOT.TFile("../../events/" + args.data + "/TTree/HH4b.root", "RECREATE")

sig_tree = sig_file.Get("Tree")
new_sig_tree = sig_tree.CloneTree(0)

w = array('f', [0])
new_sig_tree.SetBranchAddress('weight', w)

bbbb_file = ROOT.TFile("../../events/" + args.data + "/TTree/bbbb.root", 'READ')
bbbb_tree = bbbb_file.Get("Tree")

#### Get 4b norms

SR_norm = 0
CR_norm = 0
SB_norm = 0
for e in range(bbbb_tree.GetEntries()):
    bbbb_tree.GetEntry(e)

    if bbbb_tree.SR == 1:
        SR_norm += bbbb_tree.weight

    if bbbb_tree.CR == 1:
        CR_norm += bbbb_tree.weight

    if bbbb_tree.SB == 1:
        SB_norm += bbbb_tree.weight

### Get signal norms

sig_SR_norm = 0
sig_CR_norm = 0
sig_SB_norm = 0
for e in range(sig_tree.GetEntries()):
    sig_tree.GetEntry(e)

    if sig_tree.SR == 1:
        sig_SR_norm += sig_tree.weight

    if sig_tree.CR == 1:
        sig_CR_norm += sig_tree.weight

    if sig_tree.SB == 1:
        sig_SB_norm += sig_tree.weight

### Renormalize signal

for e in range(sig_tree.GetEntries()):
    sig_tree.GetEntry(e)

    if sig_tree.SR == 1:
        w[0]= sig_tree.weight * SR_norm * SR_sig_fraction / sig_SR_norm

    if sig_tree.CR == 1:
        w[0]= sig_tree.weight * CR_norm * CR_sig_fraction / sig_CR_norm

    if sig_tree.SB == 1:
        w[0]= sig_tree.weight * SB_norm * SB_sig_fraction / sig_SB_norm

    new_sig_tree.Fill()

out_file.Write()
out_file.Close()
sig_file.Close()
bbbb_file.Close()

