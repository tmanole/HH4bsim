import ROOT
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin',  default='../../events/MG1/TTree/bbjj.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG1/TTree/bbjj_small.root', type=str, help='Path to output directory.')
parser.add_argument('-N',  '--N', default=10000, type=int, help='Number of entries to select.')

args = parser.parse_args()

tree_file = ROOT.TFile(args.pathin, "READ")
tree = tree_file.Get("Tree")

new_file = ROOT.TFile(args.pathout, 'RECREATE')
new_tree = tree.CloneTree(args.N)

new_file.Write()
new_file.Close()


