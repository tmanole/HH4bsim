import ROOT
import numpy as np
from array import array
import argparse

parser = argparse.ArgumentParser(description='')
#parser.add_argument('-t', '--tpath', default='../../results/MG2/horizontal/R2/R2-R0_4c/1NN/fit.root', type=str, help='Path to TTree.')
#parser.add_argument('-o', '--outpath', default='../../results/MG2/horizontal/R2/R2-R0_4c/1NN/fit_w.root', type=str, help='Path to TTree.')
parser.add_argument('-t', '--tpath', default='../../results/MG2/horizontal/R2/R2-R0_4c/1NN/fit.root', type=str, help='Path to TTree.')
parser.add_argument('-o', '--outpath', default='../../results/MG2/horizontal/R2/R2-R0_4c/1NN/fit_w.root', type=str, help='Path to TTree.')
parser.add_argument('-N',  '--N', default=10000, type=int, help='Number of entries to select.')

args = parser.parse_args()

tree_file = ROOT.TFile(args.tpath, "WRITE")
tree = tree_file.Get("Tree")


out_file = ROOT.TFile(args.outpath, "RECREATE")
new_tree = tree.CloneTree(0)

w = array('f',[0])
new_tree.SetBranchAddress('weight', w)

N = tree.GetEntries()
pi = 7134/N

we = 0

for i in range(N):
    tree.GetEntry(i)

    w[0] = tree.weight * pi

    we += tree.weight * pi

    new_tree.Fill()

print(we)

out_file.Write()
out_file.Close()
tree_file.Close()

