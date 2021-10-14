import ROOT
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin',  default='../../events/MG2/TTree/bbbb.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG2/TTree/bbbb_tree_', type=str, help='Path to output directory.')

args = parser.parse_args()

tree_file = ROOT.TFile(args.pathin, "READ")
tree = tree_file.Get("Tree")


SR_file = ROOT.TFile(args.pathout + "SR.root", 'RECREATE')
SR_tree = tree.CloneTree(0)

for e in range(tree.GetEntries()):
    tree.GetEntry(e)

    if tree.SR == 1:
        SR_tree.Fill()

SR_file.Write()
SR_file.Close()


CR_file = ROOT.TFile(args.pathout + "CR.root", 'RECREATE')
CR_tree = tree.CloneTree(0)

for e in range(tree.GetEntries()):
    tree.GetEntry(e)

    if tree.CR == 1:
        CR_tree.Fill()

CR_file.Write()
CR_file.Close()


SB_file = ROOT.TFile(args.pathout + "SB.root", 'RECREATE')
SB_tree = tree.CloneTree(0)

for e in range(tree.GetEntries()):
    tree.GetEntry(e)

    if tree.SB == 1:
        SB_tree.Fill()

SB_file.Write()
SB_file.Close()






